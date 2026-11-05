import json
import h5py
import numpy as np
import pytest

from scripts.collect_random_data import collect, parse_args, resolve_config


IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIO = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


class ReusingEnv:
    """Observations alias mutable buffers, including across resets."""
    action_spec = (np.array([-0.25, -2.0]), np.array([0.75, 3.0]))

    def __init__(self, done_after=2, fail_after=None):
        self.done_after = done_after
        self.fail_after = fail_after
        self.episode = 0
        self.actions = []
        self.obs = {k: np.zeros((4, 4, 3), dtype=np.uint8) for k in IMAGE_KEYS}
        self.obs.update({k: np.zeros(n) for k, n in zip(PROPRIO, (3, 4, 2))})

    def fill(self, value):
        for array in self.obs.values():
            array.fill(value)
        return self.obs

    def reset(self):
        self.episode += 1
        self.step_count = 0
        return self.fill(self.episode * 10)

    def step(self, action):
        self.step_count += 1
        if self.fail_after == len(self.actions):
            raise RuntimeError("simulated failure")
        self.actions.append(action.copy())
        action[:] = 99  # Must not corrupt recorded actions.
        return self.fill(self.episode * 10 + self.step_count), 0.5, self.step_count == self.done_after, {}


def run(env, path, **overrides):
    options = dict(episodes=2, steps_per_episode=3, seed=7,
                   image_keys=IMAGE_KEYS, proprio_keys=PROPRIO, image_shape=(4, 4, 3),
                   env_args={"env_name": "Fake", "env_kwargs": {}}, provenance={})
    options.update(overrides)
    return collect(env, path, **options)


def test_alignment_mutable_buffers_and_episode_boundaries(tmp_path):
    env = ReusingEnv()
    path = run(env, tmp_path / "data.hdf5")
    assert not path.with_name(path.name + ".part").exists()
    with h5py.File(path) as f:
        assert f.attrs["collection_complete"]
        assert f["data"].attrs["total"] == 4
        assert json.loads(f.attrs["collection_config"])["seed"] == 7
        for i in range(2):
            demo = f[f"data/demo_{i}"]
            assert demo.attrs["num_samples"] == 2
            for key in (*IMAGE_KEYS, *PROPRIO):
                np.testing.assert_array_equal(demo[f"obs/{key}"][:, 0].reshape(2, -1)[:, 0], [10*(i+1), 10*(i+1)+1])
                np.testing.assert_array_equal(demo[f"next_obs/{key}"][:, 0].reshape(2, -1)[:, 0], [10*(i+1)+1, 10*(i+1)+2])
            np.testing.assert_array_equal(demo["actions"][:], env.actions[i*2:i*2+2])
            assert (demo["actions"][:] >= env.action_spec[0]).all()
            assert (demo["actions"][:] <= env.action_spec[1]).all()
            np.testing.assert_array_equal(demo["dones"][:], [False, True])
            np.testing.assert_array_equal(demo["env_dones"][:], [False, True])
            assert not demo["collection_truncated"][:].any()
            assert demo["obs"][IMAGE_KEYS[0]].dtype == np.uint8


def test_step_limit_is_distinct_from_environment_done(tmp_path):
    path = run(ReusingEnv(done_after=99), tmp_path / "data.hdf5", episodes=1)
    with h5py.File(path) as f:
        demo = f["data/demo_0"]
        assert not demo["env_dones"][:].any()
        np.testing.assert_array_equal(demo["collection_truncated"][:], [False, False, True])
        np.testing.assert_array_equal(demo["dones"][:], [False, False, True])


def test_seed_repeats_actions_and_changes_with_different_seed(tmp_path):
    actions = []
    for i, seed in enumerate((10, 10, 11)):
        with h5py.File(run(ReusingEnv(), tmp_path / f"{i}.hdf5", seed=seed)) as f:
            actions.append(f["data/demo_0/actions"][:])
    np.testing.assert_array_equal(actions[0], actions[1])
    assert not np.array_equal(actions[0], actions[2])


@pytest.mark.parametrize("partial", [False, True])
def test_existing_files_are_never_overwritten(tmp_path, partial):
    path = tmp_path / "existing.hdf5"
    occupied = path.with_name(path.name + ".part") if partial else path
    occupied.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        run(ReusingEnv(), path)
    assert occupied.read_bytes() == b"keep me"


def test_failure_preserves_committed_episodes_without_publishing(tmp_path):
    path = tmp_path / "interrupted.hdf5"
    with pytest.raises(RuntimeError, match="simulated failure"):
        run(ReusingEnv(fail_after=2), path)
    assert not path.exists()
    with h5py.File(path.with_name(path.name + ".part")) as f:
        assert not f.attrs["collection_complete"]
        assert f["data"].attrs["total"] == 2
        assert list(f["data"]) == ["demo_0"]


def test_missing_observations_fail_clearly(tmp_path):
    env = ReusingEnv()
    del env.obs[PROPRIO[0]]
    with pytest.raises(ValueError, match="missing required observation"):
        run(env, tmp_path / "bad.hdf5")


@pytest.mark.parametrize("key,value", [
    (IMAGE_KEYS[0], np.zeros((4, 4, 3), dtype=np.float32)),
    (IMAGE_KEYS[0], np.zeros((3, 4, 3), dtype=np.uint8)),
    (PROPRIO[0], np.array([np.nan, 0, 0])),
])
def test_invalid_observations_fail_before_stepping(tmp_path, key, value):
    env = ReusingEnv()
    env.reset = lambda: {**env.obs, key: value}
    with pytest.raises(ValueError, match="expected"):
        run(env, tmp_path / "bad.hdf5")
    assert not env.actions


@pytest.mark.parametrize("bounds", [([0], [np.inf]), ([1], [0]), ([0, 1], [2])])
def test_invalid_action_bounds_fail_without_creating_output(tmp_path, bounds):
    env = ReusingEnv()
    env.action_spec = bounds
    with pytest.raises(ValueError, match="action bounds"):
        run(env, tmp_path / "bad.hdf5")
    assert not list(tmp_path.iterdir())


def test_reference_metadata_and_actual_camera_resolution(tmp_path):
    reference = tmp_path / "reference.hdf5"
    with h5py.File(reference, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "NutAssemblySquare", "env_kwargs": {
            "robots": ["Panda"], "control_freq": 20, "camera_heights": 128,
            "camera_widths": 128, "controller_configs": {"type": "BASIC"}}})
        obs = data.create_group("demo_0/obs")
        for key in IMAGE_KEYS:
            obs.create_dataset(key, shape=(1, 84, 84, 3), dtype=np.uint8)
    args = parse_args(["--reference-dataset", str(reference), "--output", str(tmp_path / "out.hdf5")])
    metadata, kwargs, source = resolve_config(args)
    assert kwargs["camera_heights"] == kwargs["camera_widths"] == 84
    assert kwargs["controller_configs"] == {"type": "BASIC"}
    assert kwargs["control_freq"] == 20
    assert not kwargs["has_renderer"]
    assert kwargs["has_offscreen_renderer"] and kwargs["use_camera_obs"]
    assert metadata["env_kwargs"]["camera_heights"] == 128  # Source not mutated.
    assert len(source["sha256"]) == 64


def test_json_environment_configuration(tmp_path):
    config = tmp_path / "env.json"
    config.write_text(json.dumps({"env_name": "Lift", "env_kwargs": {"robots": "Panda"}}))
    args = parse_args(["--env-config", str(config), "--output", str(tmp_path / "out.hdf5"),
                       "--image-height", "32", "--image-width", "48"])
    metadata, kwargs, _ = resolve_config(args)
    assert metadata["env_name"] == "Lift"
    assert (kwargs["camera_heights"], kwargs["camera_widths"]) == (32, 48)


@pytest.mark.parametrize("arguments", [["--episodes", "0"], ["--steps-per-episode", "-1"], ["--seed", "-1"]])
def test_invalid_cli_limits(arguments):
    with pytest.raises(SystemExit):
        parse_args(["--env-config", "unused.json", "--output", "unused.hdf5", *arguments])
