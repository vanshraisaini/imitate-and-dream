"""True-future ranking: is the progress head g a good enough score to rank candidate chunks
if the world model were perfect?

At a decision point along a closed-loop policy run we
  1. save the simulator state and sample K action chunks from the frozen policy
     (candidate 0 is the "default" sample, as in Algorithm 1 of the report),
  2. for each candidate: restore, execute the full chunk in the simulator, encode the REAL
     observation after every step with the frozen encoder, and score it with g
     -> this is what a perfect latent dynamics model would give the ranker,
  3. for each candidate: restore, execute its first EXECUTION_HORIZON actions (what deployment
     executes), then let the default policy continue until success or the step budget
     -> the candidate's outcome.

If picking argmax(score) does not beat the default candidate here, no learned dynamics model
can make ranking work with this g; fix the score first. If it does, this is the upper bound the
learned ensemble should approach.

Usage:
    python -m scripts.world_model.true_future_ranking run --num-seeds 20 --k 8
    python -m scripts.world_model.true_future_ranking analyze

    # only seeds where the policy sometimes succeeds and sometimes fails (from a pass@K run);
    # the others cannot be helped by ranking anyway
    python -m scripts.world_model.true_future_ranking run --seed-filter mixed \
        --pass-at-k-dir results/pass_at_k_iteration2

Sharding works like test_oracle_success.py: --worker w --num-workers N, then `analyze` merges.
Each (seed, decision) is seeded on its own, so results do not depend on the number of workers.
"""

import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch

from models.world_model.latent_encoder import LatentEncoder, LatentEncoderConfig
from models.world_model.progress_head import ProgressHead
from scripts.imitation_policy.test_oracle_success import (
    DEVICE,
    EXECUTION_HORIZON,
    NUM_EVAL_STEPS,
    NUM_FLOW_STEPS,
    check_reset_seeding,
    create_environment,
    load_model,
    preprocess_observation,
    read_records,
    reset_env,
)
from scripts.imitation_policy.test_oracle_success import result_files as pass_at_k_files

POLICY_CKPT = "checkpoints/imitation_policy/nutAssemblyPolicy.pt"
ENCODER_CKPT = "checkpoints/encoder/wm_encoder_best.pt"
OUT_DIR = "results/true_future_ranking"
BASE_SEED = 777

# score variants computed from g along the true future, g[0] = current obs, g[h] = after h actions
SCORES = {
    "max": lambda g: g[1:].max(),            # Eq. 7: progress anywhere in the chunk
    "final": lambda g: g[-1],                # after the full chunk
    "exec": lambda g: g[EXECUTION_HORIZON],  # after the part that is actually executed
    "mean": lambda g: g[1:].mean(),
}


# ------------------------------------------------------------------ models
def load_encoder(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    enc = LatentEncoder(LatentEncoderConfig(**ckpt["encoder_config"]))
    missing, unexpected = enc.load_state_dict(ckpt["encoder"], strict=False)
    missing = [k for k in missing if not k.startswith("tokenizer.")]  # DINOv2 reloads from torch.hub
    if missing or unexpected:
        raise RuntimeError(f"encoder checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    head = ProgressHead(ckpt["encoder_config"]["latent_dim"], ckpt["head_hidden"])
    head.load_state_dict(ckpt["progress_head"])
    return enc.to(DEVICE).eval(), head.to(DEVICE).eval()


@torch.no_grad()
def progress_of(encoder, head, obs_list):
    """List of raw env observations -> g(E(o)) as a numpy array (len(obs_list),)."""
    obs = [preprocess_observation(o) for o in obs_list]
    batch = {k: torch.cat([o[k] for o in obs]) for k in obs[0]}
    return head.predict(encoder(batch)).float().cpu().numpy()


@torch.no_grad()
def sample_chunks(policy, obs, k):
    """K chunks for the same observation, each from its own noise. (K, H, A) numpy."""
    o = preprocess_observation(obs)
    o = {key: v.expand(k, *v.shape[1:]) for key, v in o.items()}
    return policy.predict(o, num_steps=NUM_FLOW_STEPS).cpu().numpy()


@torch.no_grad()
def policy_chunk(policy, obs):
    return sample_chunks(policy, obs, 1)[0]


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------ simulator snapshots
def snapshot(env):
    data = env.sim.data
    return {
        "state": env.sim.get_state().flatten().copy(),
        "warmstart": np.array(data.qacc_warmstart).copy(),
        "ctrl": np.array(data.ctrl).copy(),
        "timestep": env.timestep,
        "cur_time": env.cur_time,
        "controllers": [{k: copy.deepcopy(v) for k, v in vars(part).items() if k.startswith(GOAL_ATTRS)}
                        for part in _part_controllers(env)],
        # observables render on their own timers; restore them so images are taken at the same substeps
        "observables": {name: {k: copy.deepcopy(getattr(ob, k)) for k in OBS_ATTRS}
                        for name, ob in env._observables.items()},
    }


OBS_ATTRS = ("_time_since_last_sample", "_current_delay", "_sampled")


GOAL_ATTRS = ("goal", "_goal", "last_", "torques", "vels")  # controller state carried between steps


def _part_controllers(env):
    return [part for robot in env.robots for part in robot.composite_controller.part_controllers.values()]


def restore(env, snap):
    """Back to a snapshot; returns fresh observations (images re-rendered)."""
    env.sim.set_state_from_flattened(snap["state"])
    env.sim.data.qacc_warmstart[:] = snap["warmstart"]
    env.sim.data.ctrl[:] = snap["ctrl"]
    env.sim.forward()
    env.timestep, env.cur_time = snap["timestep"], snap["cur_time"]
    for part, saved in zip(_part_controllers(env), snap["controllers"]):
        for k, v in saved.items():
            setattr(part, k, copy.deepcopy(v))
    for name, saved in snap["observables"].items():
        for k, v in saved.items():
            setattr(env._observables[name], k, copy.deepcopy(v))
    return env._get_observations(force_update=True)


def check_restore_determinism(env, policy):
    """Fail fast if restore + the same actions does not reproduce the same state and images."""
    obs = reset_env(env, 0)
    seed_torch(0)
    for a in policy_chunk(policy, obs)[:8]:  # get off the initial state
        obs, *_ = env.step(a)
    snap = snapshot(env)
    actions = policy_chunk(policy, obs)

    results = []
    for _ in range(2):
        obs = restore(env, snap)
        for a in actions:
            obs, *_ = env.step(a)
        results.append((env.sim.get_state().flatten().copy(), obs["agentview_image"].copy()))
    state_err = np.abs(results[0][0] - results[1][0]).max()
    img_err = np.abs(results[0][1].astype(int) - results[1][1].astype(int)).max()
    if state_err > 1e-6 or img_err > 0:
        raise RuntimeError(f"restore is not deterministic: state diff {state_err:.2e}, image diff {img_err}")
    print(f"Restore determinism check passed (state diff {state_err:.1e}).")


# ------------------------------------------------------------------ rollouts
def step_until(env, actions, max_steps):
    """Execute actions; stop on success or budget. Returns (obs list after each step, success)."""
    obs_list, success = [], False
    for a in actions:
        if env.timestep >= max_steps:
            break
        obs, _, done, _ = env.step(a)
        obs_list.append(obs)
        success = bool(env._check_success())
        if success or done:
            break
    return obs_list, success


def continue_policy(env, policy, obs, max_steps):
    """Closed-loop default policy from the current state. Returns success."""
    while env.timestep < max_steps:
        obs_list, success = step_until(env, policy_chunk(policy, obs)[:EXECUTION_HORIZON], max_steps)
        if success:
            return True
        if not obs_list:
            break
        obs = obs_list[-1]
    return False


def evaluate_decision(env, policy, encoder, head, obs, args, seed, replan):
    """Branch K candidates at the current state. Returns (record, chunks)."""
    snap = snapshot(env)
    seed_torch(BASE_SEED + seed * 100003 + replan * 101)
    chunks = sample_chunks(policy, obs, args.k)

    g_curves, chunk_success, outcomes = [], [], []
    for k, chunk in enumerate(chunks):
        # score: g along the TRUE future of the full chunk
        obs0 = restore(env, snap)
        obs_list, succ = step_until(env, chunk, args.max_steps)
        obs_list = obs_list + [obs_list[-1]] * (len(chunk) - len(obs_list))  # pad if it ended early
        g_curves.append(progress_of(encoder, head, [obs0] + obs_list).tolist())
        chunk_success.append(succ)

        # outcome: execute what deployment executes, then the default policy takes over
        runs = []
        for r in range(args.continuations):
            obs_k = restore(env, snap)
            seed_torch(BASE_SEED + seed * 100003 + replan * 101 + 7919 * (k * args.continuations + r + 1))
            obs_list, succ = step_until(env, chunk[:EXECUTION_HORIZON], args.max_steps)
            if not succ and obs_list:
                succ = continue_policy(env, policy, obs_list[-1], args.max_steps)
            runs.append(bool(succ))
        outcomes.append(runs)

    restore(env, snap)
    rec = {"seed": seed, "replan": replan, "t": snap["timestep"], "k": args.k,
           "g": g_curves, "chunk_success": chunk_success, "success": outcomes}
    return rec, chunks


def run_seed(env, policy, encoder, head, seed, args, done_keys, out_f, log):
    """One trunk episode following the default candidate; branch at the requested replans."""
    obs = reset_env(env, seed)
    seed_torch(BASE_SEED + seed)
    decisions = sorted(set(args.decision_replans))
    for replan in range(NUM_EVAL_STEPS):
        if replan > decisions[-1] or env.timestep >= args.max_steps:
            return
        if replan in decisions and (seed, replan) not in done_keys:
            t0 = time.time()
            rec, chunks = evaluate_decision(env, policy, encoder, head, obs, args, seed, replan)
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            p = np.mean(rec["success"], axis=1)
            g = np.array([SCORES["max"](np.array(c)) for c in rec["g"]])
            log(f"seed {seed:3d} replan {replan:3d} (t={rec['t']:3d}) | outcomes {''.join('X' if x > .5 else '.' for x in p)}"
                f" | pick(max g)={p[g.argmax()]:.0f} default={p[0]:.0f} | {time.time() - t0:.0f}s")
            chunk = chunks[0]  # trunk follows the default candidate
            seed_torch(BASE_SEED + seed * 7 + replan + 1)  # trunk noise independent of branching
        else:
            chunk = policy_chunk(policy, obs)
        obs_list, success = step_until(env, chunk[:EXECUTION_HORIZON], args.max_steps)
        if success or not obs_list:
            return  # nothing left to rank
        obs = obs_list[-1]


# ------------------------------------------------------------------ commands
def select_seeds(args):
    seeds = list(range(args.num_seeds))
    if args.seed_filter != "all":
        per_seed = {}
        for (s, _), r in read_records_from(args.pass_at_k_dir).items():
            per_seed.setdefault(s, []).append(r["success"])
        rate = {s: np.mean(v) for s, v in per_seed.items()}
        keep = {"mixed": lambda x: 0 < x < 1, "not-perfect": lambda x: x < 1}[args.seed_filter]
        seeds = [s for s in sorted(rate) if keep(rate[s])][:args.num_seeds]
        print(f"{len(seeds)} '{args.seed_filter}' seeds from {args.pass_at_k_dir}")
    return seeds[args.worker::args.num_workers]


def read_records_from(pass_dir):
    import scripts.imitation_policy.test_oracle_success as oracle
    old, oracle.OUT_DIR = oracle.OUT_DIR, pass_dir
    try:
        return read_records(pass_at_k_files())
    finally:
        oracle.OUT_DIR = old


def result_files(out_dir):
    return sorted(glob.glob(os.path.join(out_dir, "decisions_w*.jsonl")))


def load_decisions(out_dir):
    recs = {}
    for path in result_files(out_dir):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # truncated last line
                    continue
                recs[(r["seed"], r["replan"])] = r
    return recs


def cmd_run(args):
    if not 0 <= args.worker < args.num_workers:
        raise SystemExit(f"--worker must be in [0, {args.num_workers - 1}]")
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = select_seeds(args)
    done_keys = set(load_decisions(args.out_dir))
    print(f"[worker {args.worker}/{args.num_workers}] {len(seeds)} seeds, K={args.k}, "
          f"decisions at replans {args.decision_replans}, budget {args.max_steps} steps")

    policy = load_model(args.policy)
    encoder, head = load_encoder(args.encoder)
    env = create_environment()
    out_path = os.path.join(args.out_dir, f"decisions_w{args.worker}.jsonl")
    try:
        check_reset_seeding(env)
        check_restore_determinism(env, policy)
        with open(out_path, "a") as f:
            for seed in seeds:
                run_seed(env, policy, encoder, head, seed, args, done_keys, f,
                         lambda m: print(f"[w{args.worker}] {m}", flush=True))
    finally:
        env.close()


def rankdata(x):
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, dtype=float)
    order = x.argsort(kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        ranks[m] = ranks[m].mean()
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a) - (len(a) - 1) / 2, rankdata(b) - (len(b) - 1) / 2
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


def pairwise_accuracy(score, p):
    """Over candidate pairs with different outcomes: how often the higher-scored one did better."""
    agree, total = 0.0, 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] == p[j]:
                continue
            total += 1
            ds = score[i] - score[j]
            agree += 0.5 if ds == 0 else float((ds > 0) == (p[i] > p[j]))
    return agree, total


def auroc(score, label):
    label = np.asarray(label, bool)
    n_pos, n_neg = label.sum(), (~label).sum()
    if not n_pos or not n_neg:
        return np.nan
    r = rankdata(score)
    return float((r[label].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def bootstrap_ci(vals, n=2000, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return np.percentile(boots, [2.5, 97.5])


def cmd_analyze(args):
    recs = load_decisions(args.out_dir)
    if not recs:
        print(f"No results in {args.out_dir}")
        return
    groups = [recs[k] for k in sorted(recs)]
    P = [np.mean(r["success"], axis=1) for r in groups]           # per-candidate success rate
    G = [np.array(r["g"]) for r in groups]                         # (K, H + 1)
    rankable = np.array([p.min() != p.max() for p in P])
    n_seeds = len({r["seed"] for r in groups})
    print(f"{len(groups)} decision points from {n_seeds} seeds, K={groups[0]['k']}, "
          f"{len(groups[0]['success'][0])} continuation(s) per candidate")
    print(f"rankable (candidates differ in outcome): {rankable.sum()} ({rankable.mean():.0%})\n")

    default = np.array([p[0] for p in P])
    rand = np.array([p.mean() for p in P])
    oracle = np.array([p.max() for p in P])

    def row(name, vals, mask):
        v = vals[mask]
        lo, hi = bootstrap_ci(v) if len(v) > 1 else (np.nan, np.nan)
        return f"  {name:<20} {v.mean():.3f}  [{lo:.3f}, {hi:.3f}]"

    for title, mask in (("all decision points", np.ones(len(groups), bool)), ("rankable only", rankable)):
        if not mask.any():
            continue
        print(f"success rate, {title} (n={mask.sum()}), 95% CI bootstrap over decisions")
        print(row("default (K=1)", default, mask))
        print(row(f"random of K", rand, mask))
        for name, fn in SCORES.items():
            pick = np.array([p[np.argmax([fn(c) for c in g])] for p, g in zip(P, G)])
            print(row(f"pick by g_{name}", pick, mask))
        print(row("oracle of K", oracle, mask))
        print()

    print("ranking quality on rankable decisions")
    print(f"  {'score':<10} {'spearman':>9} {'pair acc':>9} {'pooled AUROC':>13} {'centered AUROC':>15}")
    for name, fn in SCORES.items():
        rhos, agree, total, s_all, s_cent, y_all = [], 0.0, 0, [], [], []
        for p, g, rk in zip(P, G, rankable):
            s = np.array([fn(c) for c in g])
            s_all += s.tolist()
            s_cent += (s - s.mean()).tolist()  # removes "this state is easy/hard" from the score
            y_all += (p > 0.5).tolist()
            if rk:
                rho = spearman(s, p)
                if not np.isnan(rho):
                    rhos.append(rho)
                a, t = pairwise_accuracy(s, p)
                agree, total = agree + a, total + t
        print(f"  g_{name:<8} {np.mean(rhos) if rhos else np.nan:>9.3f} "
              f"{agree / total if total else np.nan:>9.3f} {auroc(s_all, y_all):>13.3f} "
              f"{auroc(s_cent, y_all):>15.3f}")
    print("  (chance level: spearman 0, pair acc 0.5, AUROC 0.5; pooled AUROC mixes in "
          "between-state differences)\n")

    print("by decision point (rankable / total, mean spread of g_max across candidates)")
    for replan in sorted({r["replan"] for r in groups}):
        idx = [i for i, r in enumerate(groups) if r["replan"] == replan]
        spread = np.mean([np.ptp([SCORES["max"](c) for c in G[i]]) for i in idx])
        gnow = np.mean([G[i][0, 0] for i in idx])
        print(f"  replan {replan:3d} (t={replan * EXECUTION_HORIZON:3d}): {rankable[idx].sum():3d} / {len(idx):3d}"
              f" | default {default[idx].mean():.2f} oracle {oracle[idx].mean():.2f}"
              f" | g(now) {gnow:.2f}, g_max spread {spread:.3f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run")
    p.add_argument("--policy", default=POLICY_CKPT)
    p.add_argument("--encoder", default=ENCODER_CKPT)
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--seed-filter", choices=["all", "mixed", "not-perfect"], default="all",
                   help="restrict to seeds by their pass@K per-seed success rate")
    p.add_argument("--pass-at-k-dir", default="results/pass_at_k_iteration2")
    p.add_argument("--k", type=int, default=8, help="candidates per decision")
    p.add_argument("--continuations", type=int, default=1,
                   help="policy continuations per candidate (outcome = success fraction)")
    p.add_argument("--decision-replans", type=lambda s: [int(x) for x in s.split(",")],
                   default=[0, 3, 6, 9, 12, 15],
                   help=f"replan indices to branch at (step = replan * {EXECUTION_HORIZON})")
    p.add_argument("--max-steps", type=int, default=NUM_EVAL_STEPS * EXECUTION_HORIZON,
                   help="episode step budget, same as the pass@K oracle by default")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("analyze")
    p.set_defaults(fn=cmd_analyze)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.fn(args)
