"""pass@K oracle diagnostic for a frozen flow-matching policy on robosuite Square.

For each fixed start-state seed, run the closed-loop policy N times (same initial state,
different sampling noise). A seed counts as an oracle success at K if any of K runs succeeds;
we use the unbiased pass@K estimator so one set of N runs gives the whole curve.

Usage:
    python -m scripts.imitation_policy.test_oracle_success run --num-seeds 10 --runs-per-seed 5
    python -m scripts.imitation_policy.test_oracle_success analyze

Sharding: split the seeds across N independent processes (each has its own env and policy copy).
Worker w handles seeds w, w+N, w+2N, ... and writes episodes_w<w>.jsonl. Launch all workers, e.g.
    for w in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=0 python -m scripts.imitation_policy.test_oracle_success run --worker $w --num-workers 4 &
    done; wait
(set CUDA_VISIBLE_DEVICES per worker to spread them over GPUs), then run `analyze` once to merge everything.
Results do not depend on the number of workers: every (seed, run) is seeded on its own.

To watch different runs of the same start state: set RENDER_GUI = True and run
    python -m scripts.imitation_policy.test_oracle_success run --num-seeds 1 --runs-per-seed 5
"""

import argparse
import glob
import hashlib
import json
import os
import random
import time
from math import comb

import h5py
import numpy as np
import robosuite as suite
import robosuite.macros as macros
import torch

from models.imitation_policy.policy import FlowMatchingPolicy

# Must be set before the environment is created (dataset images are upright / OpenCV convention).
macros.IMAGE_CONVENTION = "opencv"

# ----------------------------- config (match your deployment!) -----------------------------
CHECKPOINT_NAME = "checkpoints/imitation_policy/nutAssemblyPolicy.pt"
DATASET_PATH = "./datasets/square_image_v15.hdf5"
OUT_DIR = "results/pass_at_k"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FLOW_STEPS = 10
NUM_EVAL_STEPS = 100        # max number of replans per episode
EXECUTION_HORIZON = 8       # set to your real value (e.g. 8 for chunk 16 / execute 8)
K_LIST = [1, 2, 5, 10, 20]
BASE_NOISE_SEED = 12345
RENDER_GUI = False          # True: open the viewer and watch each run. Results then go to
                            # episodes_gui.jsonl so they don't mix with (or get skipped by) the main file.
# --------------------------------------------------------------------------------------------


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = FlowMatchingPolicy(**ckpt["config"])
    model.load_state_dict(ckpt["ema"])
    model.to(DEVICE).eval()
    return model


def create_environment():
    with h5py.File(DATASET_PATH, "r") as f:
        meta = json.loads(f["data"].attrs["env_args"])
    kwargs = dict(meta["env_kwargs"])
    kwargs.update(has_renderer=RENDER_GUI, has_offscreen_renderer=True, use_camera_obs=True)
    return suite.make(meta["env_name"], **kwargs)


def preprocess_image(image):
    img = torch.as_tensor(image.copy(), dtype=torch.float32, device=DEVICE)
    return (img.permute(2, 0, 1) / 255.0).unsqueeze(0)


def preprocess_observation(obs):
    proprio = np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]])
    return {
        "agent_image": preprocess_image(obs["agentview_image"]),
        "wrist_image": preprocess_image(obs["robot0_eye_in_hand_image"]),
        "proprio": torch.as_tensor(proprio, dtype=torch.float32, device=DEVICE).unsqueeze(0),
    }


@torch.no_grad()
def predict_action_chunk(model, obs):
    chunk = model.predict(preprocess_observation(obs), num_steps=NUM_FLOW_STEPS)
    return chunk.squeeze(0).cpu().numpy()


def _env_generators(env):
    """All numpy Generators that influence env.reset(): the env's own and those held by the placement samplers.

    robosuite builds its placement samplers once and keeps a reference to env.rng, so replacing env.rng with a
    new Generator does NOT change what the samplers draw. The existing objects must be reseeded in place.
    """
    gens, seen = [], set()

    def add(g):
        if isinstance(g, np.random.Generator) and id(g) not in seen:
            seen.add(id(g))
            gens.append(g)

    add(getattr(env, "rng", None))
    stack = [getattr(env, "placement_initializer", None)]
    while stack:
        sampler = stack.pop()
        if sampler is None:
            continue
        add(getattr(sampler, "rng", None))
        sub = getattr(sampler, "samplers", None)
        if isinstance(sub, dict):
            stack.extend(sub.values())
        elif isinstance(sub, (list, tuple)):
            stack.extend(sub)
    return gens


def reset_env(env, seed):
    """Reset to the start state determined by `seed` (same seed -> identical initial state)."""
    random.seed(seed)       # older robosuite versions draw from the global RNGs
    np.random.seed(seed)
    for gen in _env_generators(env):
        gen.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    return env.reset()


def init_fingerprint(env):
    """Hash of the full simulator state (object poses, robot joints, ...) right after reset."""
    state = np.round(env.sim.get_state().flatten(), 6)
    return hashlib.md5(state.tobytes()).hexdigest()[:12]


def check_reset_seeding(env):
    """Fail fast if resetting with the same seed does not reproduce the same initial state."""
    fps = []
    for seed in (0, 0, 1, 0):
        reset_env(env, seed)
        fps.append(init_fingerprint(env))
    if not (fps[0] == fps[1] == fps[3]):
        raise RuntimeError("Seeding is broken: resetting with the same seed gave different initial states. "
                           "pass@K would be meaningless.")
    if fps[0] == fps[2]:
        raise RuntimeError("Seeding is broken: different seeds gave the same initial state.")
    print("Reset seeding check passed (same seed -> same initial state, different seeds -> different states).")


def run_episode(env, model, seed, run_idx):
    obs = reset_env(env, seed)
    fingerprint = init_fingerprint(env)

    # Policy sampling noise: seeded separately and differently for every run.
    torch.manual_seed(BASE_NOISE_SEED + seed * 1000 + run_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_NOISE_SEED + seed * 1000 + run_idx)

    success, done = False, False
    for _ in range(NUM_EVAL_STEPS):
        chunk = predict_action_chunk(model, obs)
        for action in chunk[:EXECUTION_HORIZON]:
            obs, _, done, _ = env.step(action)
            if RENDER_GUI:
                env.render()
            success = bool(env._check_success())
            if done or success:
                break
        if done or success:
            break

    return {"seed": seed, "run": run_idx, "success": success,
            "env_steps": int(env.timestep), "init_fp": fingerprint}


def result_files():
    """All result files `analyze` merges: the unsharded episodes.jsonl (if present) and every worker file."""
    paths = [os.path.join(OUT_DIR, "episodes.jsonl")]
    paths += sorted(glob.glob(os.path.join(OUT_DIR, "episodes_w*.jsonl")))
    return [p for p in paths if os.path.exists(p)]


def read_records(paths):
    """Read episode records, keeping one record per (seed, run) even if it appears in several files."""
    records = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # truncated last line (killed mid-write, or a worker is writing now)
                    continue
                records[(r["seed"], r["run"])] = r
    return records


def cmd_run(args):
    if not 0 <= args.worker < args.num_workers:
        raise SystemExit(f"--worker must be in [0, {args.num_workers - 1}], got {args.worker}")

    os.makedirs(OUT_DIR, exist_ok=True)
    if RENDER_GUI:
        out_path = os.path.join(OUT_DIR, "episodes_gui.jsonl")
        resume_paths = [out_path]
    else:
        out_path = os.path.join(OUT_DIR, f"episodes_w{args.worker}.jsonl")
        # Skip anything finished in ANY earlier launch, whatever its --num-workers was. The seed -> worker
        # mapping changes with the worker count, so a worker's own file alone is not enough.
        resume_paths = result_files()
    done_keys = set(read_records([p for p in resume_paths if os.path.exists(p)]))

    seeds = list(range(args.num_seeds))[args.worker::args.num_workers]  # interleaved: balanced load
    print(f"[worker {args.worker}/{args.num_workers}] {len(seeds)} seeds: {seeds[:5]}{'...' if len(seeds) > 5 else ''}")
    model, env = load_model(CHECKPOINT_NAME), create_environment()

    t0, n_done = time.time(), 0
    try:
        check_reset_seeding(env)
        with open(out_path, "a") as f:
            for seed in seeds:
                for run_idx in range(args.runs_per_seed):
                    if (seed, run_idx) in done_keys:
                        continue
                    rec = run_episode(env, model, seed, run_idx)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    n_done += 1
                    print(f"[w{args.worker}] seed {seed:3d} run {run_idx:2d} "
                          f"{'SUCCESS' if rec['success'] else 'fail   '} "
                          f"({(time.time() - t0) / n_done:.1f}s/episode)")
    finally:
        env.close()


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def cmd_analyze(args):
    per_seed = {}   # seed -> list of success flags
    fps = {}        # seed -> set of init fingerprints
    paths = result_files()
    print(f"Reading {len(paths)} result file(s): {[os.path.basename(p) for p in paths]}")
    for (seed, _run), r in sorted(read_records(paths).items()):
        per_seed.setdefault(seed, []).append(r["success"])
        fps.setdefault(seed, set()).add(r["init_fp"])

    if not per_seed:
        print("No results found.")
        return

    bad = [s for s, v in fps.items() if len(v) > 1]
    if bad:
        print(f"WARNING: initial state differs across runs for seeds {bad[:10]}"
              f"{'...' if len(bad) > 10 else ''} -> seeding of env.reset is not working; pass@K is invalid.")

    seeds = sorted(per_seed)
    n_min = min(len(per_seed[s]) for s in seeds)
    ns = np.array([len(per_seed[s]) for s in seeds])
    cs = np.array([sum(per_seed[s]) for s in seeds])
    rates = cs / ns
    print(f"\n{len(seeds)} seeds, {n_min}-{ns.max()} runs per seed")

    rng = np.random.default_rng(0)
    print(f"\n{'K':>4} | {'pass@K':>7} | 95% CI (bootstrap over seeds)")
    print("-" * 42)
    curve = {}
    for k in K_LIST:
        if k > n_min:
            continue
        vals = np.array([pass_at_k(n, c, k) for n, c in zip(ns, cs)])
        boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        curve[k] = vals.mean()
        print(f"{k:>4} | {vals.mean():>7.3f} | [{lo:.3f}, {hi:.3f}]")

    n_zero, n_one = int((rates == 0).sum()), int((rates == 1).sum())
    n_mid = len(rates) - n_zero - n_one
    print(f"\nPer-seed success rate: always fail {n_zero}, always succeed {n_one}, "
          f"mixed {n_mid} ({n_mid / len(rates):.0%}) <- only mixed seeds are rankable")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(list(curve), list(curve.values()), marker="o")
        # ax[0].set_xscale("log")
        ax[0].set_xlabel("K")
        ax[0].set_ylabel("pass@K (oracle success)")
        ax[0].set_ylim(0, 1)
        ax[0].grid(alpha=0.3)
        ax[1].hist(rates, bins=np.linspace(0, 1, 11), edgecolor="black")
        ax[1].set_xlabel("per-seed success rate")
        ax[1].set_ylabel("number of seeds")
        fig.tight_layout()
        fig_path = os.path.join(OUT_DIR, "pass_at_k.png")
        fig.savefig(fig_path, dpi=150)
        print(f"Saved plot to {fig_path}")
    except ImportError:
        print("matplotlib not installed; skipping plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--num-seeds", type=int, default=100)
    p_run.add_argument("--runs-per-seed", type=int, default=20)
    p_run.add_argument("--worker", type=int, default=0, help="index of this worker (0-based)")
    p_run.add_argument("--num-workers", type=int, default=1, help="total number of worker processes")
    p_run.set_defaults(fn=cmd_run)

    p_an = sub.add_parser("analyze")
    p_an.set_defaults(fn=cmd_analyze)

    args = parser.parse_args()
    args.fn(args)