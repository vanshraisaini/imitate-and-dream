import json

import h5py
import numpy as np
import pytest

from scripts.datasets.collect_policy_data import NOISE_SEED_OFFSET, collect, parse_args

IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIO = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
STATE = {"object": "object-state"}


class FakeEnv:
    """Observations alias one mutable buffer; success after `success_after` steps (None = never)."""

    def __init__(self, success_after=None):
        self.success_after = success_after
        self.obs = {k: np.zeros((4, 4, 3), dtype=np.uint8) for k in IMAGE_KEYS}
        self.obs.update({k: np.zeros(n) for k, n in zip(PROPRIO, (3, 4, 2))})
        self.obs["object-state"] = np.zeros(14)
        self.seeds, self.actions = [], []

    def fill(self, value):
        for array in self.obs.values():
            array.fill(value)
        return self.obs

    def reset(self, seed):
        self.seeds.append(seed)
        self.t = 0
        return self.fill(0)

    def step(self, action):
        self.actions.append(action.copy())
        action[:] = 99  # must not corrupt recorded actions
        self.t += 1
        return self.fill(self.t), 0.0, False, {}

    def success(self):
        return self.success_after is not None and self.t >= self.success_after


class CountingPolicy:
    """Chunk k is filled with k (so executed actions show which chunk they came from)."""

    def __init__(self, horizon=4, action_dim=2):
        self.horizon, self.action_dim, self.calls, self.seen = horizon, action_dim, 0, []

    def __call__(self, obs):
        self.seen.append(int(obs["robot0_eef_pos"][0]))  # which observation the policy was given
        chunk = np.full((self.horizon, self.action_dim), float(self.calls))
        self.calls += 1
        return chunk


def run(env, policy, path, **overrides):
    options = dict(episodes=2, start_seed=5, max_steps=7, execution_horizon=2,
                   image_keys=IMAGE_KEYS, proprio_keys=PROPRIO, state_keys=STATE, image_shape=(4, 4, 3),
                   env_args={"env_name": "Fake", "env_kwargs": {}}, provenance={})
    options.update(overrides)
    noise = []
    path = collect(env, path, policy, env.reset, env.success, seed_noise=noise.append, **options)
    return path, noise


def test_failed_episodes_run_to_budget_with_replanning(tmp_path):
    env, policy = FakeEnv(), CountingPolicy()
    path, noise = run(env, policy, tmp_path / "data.hdf5")
    assert env.seeds == [5, 6] and noise == [NOISE_SEED_OFFSET + 5, NOISE_SEED_OFFSET + 6]
    with h5py.File(path) as f:
        assert f.attrs["collection_complete"]
        cfg = json.loads(f.attrs["collection_config"])
        assert cfg["sampler"] == "flow_matching_policy" and cfg["execution_horizon"] == 2
        assert f["data"].attrs["total"] == 14 and f["data"].attrs["num_success"] == 0
        ep = f["data/demo_0"]
        assert not ep.attrs["success"] and ep.attrs["success_step"] == -1 and ep.attrs["start_seed"] == 5
        # 7 steps at 2 actions per chunk -> chunks 0,0,1,1,2,2,3 ; 4 chunks sampled
        np.testing.assert_array_equal(ep["actions"][:, 0], [0, 0, 1, 1, 2, 2, 3])
        np.testing.assert_array_equal(ep["chunk_start"][:], [1, 0, 1, 0, 1, 0, 1])
        assert ep["action_chunks"].shape == (4, 4, 2)
        # obs / next_obs alignment despite the aliased buffer
        np.testing.assert_array_equal(ep["obs/robot0_eef_pos"][:, 0], np.arange(7))
        np.testing.assert_array_equal(ep["next_obs/robot0_eef_pos"][:, 0], np.arange(1, 8))
        assert ep["obs/object"].shape == (7, 14)
        assert not ep["success"][:].any() and not ep["env_dones"][:].any()
        np.testing.assert_array_equal(ep["dones"][:], [0, 0, 0, 0, 0, 0, 1])
        assert ep["collection_truncated"][-1]
    # policy always replans from the latest observation
    assert policy.seen[:4] == [0, 2, 4, 6]


def test_success_stops_episode_and_is_labelled(tmp_path):
    env = FakeEnv(success_after=3)
    path, _ = run(env, CountingPolicy(), tmp_path / "data.hdf5", episodes=1)
    with h5py.File(path) as f:
        ep = f["data/demo_0"]
        assert ep.attrs["success"] and ep.attrs["success_step"] == 2
        np.testing.assert_array_equal(ep["success"][:], [0, 0, 1])
        assert ep["dones"][-1] and ep["collection_truncated"][-1]
        assert f["data"].attrs["num_success"] == 1


def test_post_success_steps(tmp_path):
    env = FakeEnv(success_after=3)
    path, _ = run(env, CountingPolicy(), tmp_path / "data.hdf5", episodes=1, post_success_steps=2)
    with h5py.File(path) as f:
        ep = f["data/demo_0"]
        assert ep.attrs["success_step"] == 2 and ep.attrs["num_samples"] == 5
        np.testing.assert_array_equal(ep["success"][:], [0, 0, 1, 1, 1])


def test_refuses_overwrite(tmp_path):
    out = tmp_path / "data.hdf5"
    out.touch()
    with pytest.raises(FileExistsError):
        run(FakeEnv(), CountingPolicy(), out)


def test_parse_args(tmp_path):
    args = parse_args(["--reference-dataset", "x.hdf5", "--output", "y.hdf5", "--start-seed", "3"])
    assert args.seed == 3 and args.execution_horizon == 8 and args.max_steps == 800
    with pytest.raises(SystemExit):
        parse_args(["--reference-dataset", "x.hdf5", "--output", "y.hdf5", "--start-seed", "-1"])
