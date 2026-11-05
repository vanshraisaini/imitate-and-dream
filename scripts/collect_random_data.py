"""Collect episode-based random-action transitions from robosuite.

Run with ``python -m scripts.collect_random_data --help``. Simulator imports are
delayed so configuration, storage, and alignment can be tested without OpenGL.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random

import h5py
import numpy as np


CAMERAS = ("agentview", "robot0_eye_in_hand")
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-dataset", type=Path, help="HDF5 containing data.attrs['env_args']")
    source.add_argument("--env-config", type=Path, help="JSON with env_name and env_kwargs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--steps-per-episode", type=positive_int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-names", nargs=2, default=CAMERAS, metavar=("AGENT", "WRIST"))
    parser.add_argument("--image-height", type=positive_int)
    parser.add_argument("--image-width", type=positive_int)
    parser.add_argument("--proprio-keys", nargs="+", default=PROPRIO_KEYS)
    parser.add_argument("--render", action="store_true", help="also display an interactive viewer")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if len(set(args.camera_names)) != 2:
        parser.error("--camera-names must specify two distinct cameras")
    if len(set(args.proprio_keys)) != len(args.proprio_keys):
        parser.error("--proprio-keys must be distinct")
    if set(args.proprio_keys) & {f"{name}_image" for name in args.camera_names}:
        parser.error("--proprio-keys must not include camera image keys")
    return args


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_config(args):
    source = args.reference_dataset or args.env_config
    image_shape = None
    if args.reference_dataset:
        with h5py.File(source, "r") as f:
            metadata = json.loads(f["data"].attrs["env_args"])
            # Image observations are authoritative over possibly stale env kwargs.
            demos = list(f["data"])
            if demos and "obs" in f["data"][demos[0]]:
                obs = f["data"][demos[0]]["obs"]
                shapes = [obs[f"{name}_image"].shape[1:] for name in args.camera_names
                          if f"{name}_image" in obs]
                if shapes:
                    if len(shapes) != 2 or shapes[0] != shapes[1] or len(shapes[0]) != 3 or shapes[0][-1] != 3:
                        raise ValueError("Reference must contain both requested RGB cameras at the same resolution")
                    image_shape = shapes[0][:2]
    else:
        metadata = json.loads(source.read_text())
    if not isinstance(metadata, dict) or not isinstance(metadata.get("env_name"), str) or not isinstance(metadata.get("env_kwargs"), dict):
        raise ValueError("Environment metadata requires env_name (string) and env_kwargs (object)")
    if metadata.get("type", 1) != 1:
        raise ValueError("Only robosuite environment metadata (type 1) is supported")
    kwargs = copy.deepcopy(metadata["env_kwargs"])

    def dimension(override, key, index):
        value = override if override is not None else (image_shape[index] if image_shape else kwargs.get(key, 84))
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Specify --image-{'height' if index == 0 else 'width'} for non-scalar {key}")
        return value

    kwargs.update(has_renderer=args.render, has_offscreen_renderer=True, use_camera_obs=True,
                  camera_names=list(args.camera_names), camera_depths=False,
                  camera_heights=dimension(args.image_height, "camera_heights", 0),
                  camera_widths=dimension(args.image_width, "camera_widths", 1), seed=args.seed)
    return metadata, kwargs, {"path": str(source.resolve()), "sha256": sha256(source), "env_args": metadata}


def snapshot(obs, image_keys, proprio_keys, image_shape):
    """Copy observations before stepping: simulators may reuse array storage."""
    result = {}
    for key in (*image_keys, *proprio_keys):
        if key not in obs:
            raise ValueError(f"Environment is missing required observation {key!r}")
        value = np.array(obs[key], copy=True)
        if key in image_keys:
            if value.shape != image_shape or value.dtype != np.uint8:
                raise ValueError(f"{key}: expected uint8 RGB {image_shape}, got {value.dtype} {value.shape}")
        elif value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(f"{key}: expected a finite proprioception vector")
        result[key] = value
    return result


def collect(env, output, *, episodes, steps_per_episode, seed, image_keys,
            proprio_keys, image_shape, env_args, provenance, render=False):
    """Write bounded per-episode buffers; publish only a complete, closed HDF5.

    The caller owns the environment. Interrupted runs retain an incomplete .part
    file containing any fully written episodes; neither output nor .part is reused.
    """
    output = Path(output)
    partial = output.with_name(output.name + ".part")
    if episodes < 1 or steps_per_episode < 1:
        raise ValueError("episodes and steps_per_episode must be positive")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    low, high = (np.asarray(x, dtype=np.float64) for x in env.action_spec)
    if low.ndim != 1 or low.size == 0 or low.shape != high.shape or not np.isfinite([low, high]).all() or np.any(low > high):
        raise ValueError("Expected finite, ordered per-dimension action bounds")
    rng = np.random.default_rng(seed)
    with h5py.File(partial, "x") as f:
        f.attrs["collection_complete"] = False
        f.attrs["schema_version"] = 1
        f.attrs["collection_config"] = json.dumps({
            "sampler": "independent_uniform", "seed": seed, "episodes": episodes,
            "steps_per_episode": steps_per_episode, "image_keys": list(image_keys),
            "proprio_keys": list(proprio_keys), "image_convention": "opencv", "color_order": "RGB",
            "action_low": low.tolist(), "action_high": high.tolist(),
            "created_at": datetime.now(timezone.utc).isoformat(), **provenance,
        })
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(env_args)
        data.attrs["total"] = 0
        for episode in range(episodes):
            current = snapshot(env.reset(), image_keys, proprio_keys, image_shape)
            rows = []
            for step in range(steps_per_episode):
                action = rng.uniform(low, high)
                # Preserve the sampled action even if an environment mutates its input.
                following, reward, done, _ = env.step(action.copy())
                following = snapshot(following, image_keys, proprio_keys, image_shape)
                if not np.isfinite(reward):
                    raise ValueError("Environment returned a non-finite reward")
                truncated = step + 1 == steps_per_episode and not done
                rows.append((current, action, following, float(reward), bool(done), truncated))
                current = following
                if render:
                    env.render()
                if done:
                    break
            group = data.create_group(f"demo_{episode}")
            for index, name in ((0, "obs"), (2, "next_obs")):
                obs_group = group.create_group(name)
                for key in (*image_keys, *proprio_keys):
                    obs_group.create_dataset(key, data=np.stack([row[index][key] for row in rows]), compression="gzip")
            group.create_dataset("actions", data=np.stack([row[1] for row in rows]))
            group.create_dataset("rewards", data=[row[3] for row in rows])
            group.create_dataset("env_dones", data=[row[4] for row in rows], dtype=np.bool_)
            group.create_dataset("collection_truncated", data=[row[5] for row in rows], dtype=np.bool_)
            group.create_dataset("dones", data=[row[4] or row[5] for row in rows], dtype=np.bool_)
            group.attrs["num_samples"] = len(rows)
            data.attrs.modify("total", int(data.attrs["total"]) + len(rows))
            f.flush()
            print(f"Episode {episode + 1}/{episodes}: {len(rows)} transitions", flush=True)
        f.attrs.modify("collection_complete", True)
    # Same-directory hard link publishes atomically and refuses any existing target.
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
    import robosuite as suite
    import robosuite.macros as macros
    import mujoco

    macros.IMAGE_CONVENTION = "opencv"
    # robosuite 1.5 uses both its own Generator and legacy NumPy samplers.
    np.random.seed(args.seed)
    random.seed(args.seed)
    if metadata.get("env_version") and metadata["env_version"] != suite.__version__:
        print(f"Version difference: source robosuite {metadata['env_version']}, installed {suite.__version__}")
    env = suite.make(metadata["env_name"], **kwargs)
    try:
        output = collect(
            env, args.output, episodes=args.episodes, steps_per_episode=args.steps_per_episode,
            seed=args.seed, image_keys=tuple(f"{name}_image" for name in args.camera_names),
            proprio_keys=tuple(args.proprio_keys),
            image_shape=(kwargs["camera_heights"], kwargs["camera_widths"], 3),
            env_args={"env_name": metadata["env_name"], "type": 1,
                      "env_version": suite.__version__, "env_kwargs": kwargs},
            provenance={"source": source, "collector_sha256": sha256(__file__),
                        "versions": {"robosuite": suite.__version__, "mujoco": mujoco.__version__,
                                     "numpy": np.__version__, "h5py": h5py.__version__}}, render=args.render,
        )
    finally:
        env.close()
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
