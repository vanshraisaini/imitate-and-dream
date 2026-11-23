"""Collect episodes from the frozen flow-matching policy in robosuite, with success labels.

Same HDF5 layout as collect_random_data.py (obs / next_obs / actions / rewards / dones, plus the
privileged "object" state), with actions chosen by the policy instead of uniformly at random:
the policy predicts a chunk of `horizon` actions and the first --execution-horizon are executed
before replanning, exactly as in deployment. On top of the random layout it stores:

    data/demo_i attrs:  success (bool), success_step (int, -1 if never), start_seed, noise_seed
    data/demo_i/success        bool [T]   task success after this transition
    data/demo_i/chunk_start    bool [T]   True where a new chunk was sampled before this action
    data/demo_i/action_chunks  float32 [C, horizon, action_dim]  every sampled chunk (incl. unexecuted tail)

Episode i starts from the state that seed (--start-seed + i) gives, using the same reset seeding as
the pass@K oracle (scripts/imitation_policy/test_oracle_success.py), so episodes can be matched to
its per-seed results. To collect in parallel, run several processes with disjoint --start-seed
ranges and different --output files; EncoderWMDataset accepts several files.

Run with ``python -m scripts.datasets.collect_policy_data --help``. Simulator and torch imports are
delayed so the collection loop can be tested with a fake environment and policy.

python -m scripts.datasets.collect_policy_data \
    --reference-dataset datasets/square_image_v15.hdf5 \
    --output datasets/policy/square_policy_s0.hdf5 \
    --episodes 100 --start-seed 0
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import h5py
import numpy as np

from scripts.datasets.collect_random_data import (
    CAMERAS, PROPRIO_KEYS, STATE_KEYS, low_dim_sources, positive_int, resolve_config, sha256, snapshot,
)

DEFAULT_CHECKPOINT = "checkpoints/imitation_policy/nutAssemblyPolicy.pt"
NOISE_SEED_OFFSET = 1_000_003  # policy noise seed = offset + start seed (independent of the reset RNG)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-dataset", type=Path, help="HDF5 containing data.attrs['env_args']")
    source.add_argument("--env-config", type=Path, help="JSON with env_name and env_kwargs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--start-seed", type=int, default=0, help="episode i uses start-state seed start_seed + i")
    parser.add_argument("--max-steps", type=positive_int, default=350,
                        help="step budget per episode (800 = the pass@K oracle's 100 replans x 8)")
    parser.add_argument("--execution-horizon", type=positive_int, default=8,
                        help="actions executed from each predicted chunk before replanning")
    parser.add_argument("--flow-steps", type=positive_int, default=10, help="Euler steps per chunk")
    parser.add_argument("--post-success-steps", type=int, default=0,
                        help="keep running the policy this many steps after success (0 = stop at success)")
    parser.add_argument("--camera-names", nargs=2, default=CAMERAS, metavar=("AGENT", "WRIST"))
    parser.add_argument("--image-height", type=positive_int)
    parser.add_argument("--image-width", type=positive_int)
    parser.add_argument("--proprio-keys", nargs="+", default=PROPRIO_KEYS)
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    parser.add_argument("--render", action="store_true", help="also display an interactive viewer")
    args = parser.parse_args(argv)
    if args.start_seed < 0:
        parser.error("--start-seed must be nonnegative")
    if args.post_success_steps < 0:
        parser.error("--post-success-steps must be nonnegative")
    if len(set(args.camera_names)) != 2:
        parser.error("--camera-names must specify two distinct cameras")
    if len(set(args.proprio_keys)) != len(args.proprio_keys):
        parser.error("--proprio-keys must be distinct")
    args.seed = args.start_seed  # resolve_config seeds the env with args.seed
    return args


def write_episode(data, name, rows, chunks, image_keys, low_dim_keys, attrs):
    """rows: (obs, action, next_obs, reward, env_done, truncated, success, chunk_start) per transition."""
    group = data.create_group(name)
    for index, sub in ((0, "obs"), (2, "next_obs")):
        obs_group = group.create_group(sub)
        for key in (*image_keys, *low_dim_keys):
            obs_group.create_dataset(key, data=np.stack([row[index][key] for row in rows]), compression="gzip")
    group.create_dataset("actions", data=np.stack([row[1] for row in rows]))
    group.create_dataset("rewards", data=[row[3] for row in rows])
    group.create_dataset("env_dones", data=[row[4] for row in rows], dtype=np.bool_)
    group.create_dataset("collection_truncated", data=[row[5] for row in rows], dtype=np.bool_)
    group.create_dataset("dones", data=[row[4] or row[5] for row in rows], dtype=np.bool_)
    group.create_dataset("success", data=[row[6] for row in rows], dtype=np.bool_)
    group.create_dataset("chunk_start", data=[row[7] for row in rows], dtype=np.bool_)
    group.create_dataset("action_chunks", data=np.stack(chunks).astype(np.float32))
    group.attrs["num_samples"] = len(rows)
    for key, value in attrs.items():
        group.attrs[key] = value


def run_episode(env, policy_fn, reset_fn, success_fn, *, start_seed, max_steps, execution_horizon,
                post_success_steps, image_keys, low_dim_keys, image_shape, render=False):
    """One closed-loop policy episode. Returns (rows, chunks, success_step)."""
    raw = reset_fn(start_seed)
    current = snapshot(raw, image_keys, low_dim_keys, image_shape)
    rows, chunks, success_step, step = [], [], -1, 0
    while True:
        chunk = np.array(policy_fn(raw), dtype=np.float64)  # policy reads the env's own observation dict
        chunks.append(chunk)
        for i, action in enumerate(chunk[:execution_horizon]):
            raw, reward, done, _ = env.step(action.copy())
            following = snapshot(raw, image_keys, low_dim_keys, image_shape)
            if not np.isfinite(reward):
                raise ValueError("Environment returned a non-finite reward")
            success = bool(success_fn())
            if success and success_step < 0:
                success_step = step
            step += 1
            stop = done or step >= max_steps or (success_step >= 0 and step - success_step > post_success_steps)
            rows.append((current, action, following, float(reward), bool(done), stop and not done, success, i == 0))
            current = following
            if render:
                env.render()
            if stop:
                return rows, chunks, success_step


def collect(env, output, policy_fn, reset_fn, success_fn, *, episodes, start_seed, max_steps,
            execution_horizon, image_keys, proprio_keys, image_shape, env_args, provenance,
            post_success_steps=0, state_keys=None, render=False, seed_noise=None, policy_info=None):
    """Write policy episodes to an HDF5 file; publish only a complete, closed file.

    policy_fn(raw_obs) -> (horizon, action_dim) chunk; reset_fn(seed) -> raw_obs (seeded reset);
    success_fn() -> bool; seed_noise(seed) seeds the policy's sampling noise before each episode.
    Same output protection as collect_random_data.collect (.part file, atomic hard-link publish).
    """
    output = Path(output)
    partial = output.with_name(output.name + ".part")
    if episodes < 1 or max_steps < 1 or execution_horizon < 1:
        raise ValueError("episodes, max_steps and execution_horizon must be positive")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    low_dim_keys = low_dim_sources(proprio_keys, state_keys)
    n_success = 0
    with h5py.File(partial, "x") as f:
        f.attrs["collection_complete"] = False
        f.attrs["schema_version"] = 1
        f.attrs["collection_config"] = json.dumps({
            "sampler": "flow_matching_policy", "policy": policy_info or {},
            "start_seed": start_seed, "episodes": episodes, "max_steps": max_steps,
            "execution_horizon": execution_horizon, "post_success_steps": post_success_steps,
            "noise_seed_offset": NOISE_SEED_OFFSET, "image_keys": list(image_keys),
            "proprio_keys": list(proprio_keys), "state_keys": dict(state_keys or {}),
            "image_convention": "opencv", "color_order": "RGB",
            "created_at": datetime.now(timezone.utc).isoformat(), **provenance,
        })
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["total"] = 0
        for episode in range(episodes):
            seed = start_seed + episode
            noise_seed = NOISE_SEED_OFFSET + seed
            if seed_noise is not None:
                seed_noise(noise_seed)
            rows, chunks, success_step = run_episode(
                env, policy_fn, reset_fn, success_fn, start_seed=seed, max_steps=max_steps,
                execution_horizon=execution_horizon, post_success_steps=post_success_steps,
                image_keys=image_keys, low_dim_keys=low_dim_keys, image_shape=image_shape, render=render)
            success = success_step >= 0
            n_success += success
            write_episode(data, f"demo_{episode}", rows, chunks, image_keys, low_dim_keys,
                          {"success": success, "success_step": success_step,
                           "start_seed": seed, "noise_seed": noise_seed})
            data.attrs.modify("total", int(data.attrs["total"]) + len(rows))
            f.flush()
            print(f"Episode {episode + 1}/{episodes} (seed {seed}): {'SUCCESS' if success else 'fail   '} "
                  f"{len(rows)} transitions | success so far {n_success}/{episode + 1}", flush=True)
        data.attrs["num_success"] = n_success
        f.attrs.modify("collection_complete", True)
    os.link(partial, output)
    partial.unlink()
    return output


def main(argv=None):
    args = parse_args(argv)
    metadata, kwargs, source = resolve_config(args)
    if args.output.exists() or args.output.with_name(args.output.name + ".part").exists():
        raise FileExistsError(f"Output or partial file already exists for {args.output}")
    if not args.render:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco
    import robosuite as suite
    import robosuite.macros as macros
    import torch

    from models.imitation_policy.policy import FlowMatchingPolicy
    from scripts.imitation_policy.test_oracle_success import check_reset_seeding, reset_env

    macros.IMAGE_CONVENTION = "opencv"  # dataset images are upright
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = FlowMatchingPolicy.from_checkpoint(ckpt)
    policy.to(device).eval()

    agent_key, wrist_key = (f"{name}_image" for name in args.camera_names)

    def image(x):
        return (torch.as_tensor(x.copy(), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0)[None]

    @torch.no_grad()
    def policy_fn(obs):
        proprio = np.concatenate([obs[k] for k in args.proprio_keys])
        batch = {"agent_image": image(obs[agent_key]), "wrist_image": image(obs[wrist_key]),
                 "proprio": torch.as_tensor(proprio, dtype=torch.float32, device=device)[None]}
        return policy.predict(batch, num_steps=args.flow_steps)[0].cpu().numpy()

    def seed_noise(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if metadata.get("env_version") and metadata["env_version"] != suite.__version__:
        print(f"Version difference: source robosuite {metadata['env_version']}, installed {suite.__version__}")
    env = suite.make(metadata["env_name"], **kwargs)
    try:
        check_reset_seeding(env)
        output = collect(
            env, args.output, policy_fn, lambda seed: reset_env(env, seed), env._check_success,
            episodes=args.episodes, start_seed=args.start_seed, max_steps=args.max_steps,
            execution_horizon=args.execution_horizon, post_success_steps=args.post_success_steps,
            image_keys=(agent_key, wrist_key), proprio_keys=tuple(args.proprio_keys),
            state_keys=STATE_KEYS, image_shape=(kwargs["camera_heights"], kwargs["camera_widths"], 3),
            env_args={"env_name": metadata["env_name"], "type": 1,
                      "env_version": suite.__version__, "env_kwargs": kwargs},
            provenance={"source": source, "collector_sha256": sha256(__file__),
                        "versions": {"robosuite": suite.__version__, "mujoco": mujoco.__version__,
                                     "torch": torch.__version__, "numpy": np.__version__,
                                     "h5py": h5py.__version__}},
            render=args.render, seed_noise=seed_noise,
            policy_info={"checkpoint": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint),
                         "config": ckpt["config"], "weights": "ema", "flow_steps": args.flow_steps},
        )
    finally:
        env.close()
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
