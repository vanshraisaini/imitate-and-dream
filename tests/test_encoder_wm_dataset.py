import h5py
import numpy as np
import pytest
import torch

from models.world_model.progress_head import progress_labels
from scripts.datasets.encoder_wm_dataset import KIND_INDEX, EncoderWMDataset, Source, split_episodes


def make_file(path, lengths, outcomes=None):
    """Tiny HDF5 in the demo / collector layout. outcomes: per-episode (success, success_step) or None."""
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i, T in enumerate(lengths):
            g = data.create_group(f"demo_{i}")
            obs = g.create_group("obs")
            for k in ("agentview_image", "robot0_eye_in_hand_image"):
                obs[k] = np.full((T, 4, 4, 3), i, dtype=np.uint8)
            obs["robot0_eef_pos"] = np.zeros((T, 3))
            obs["robot0_eef_quat"] = np.tile([0.0, 0.0, 0.0, 1.0], (T, 1))
            obs["robot0_gripper_qpos"] = np.zeros((T, 2))
            obs["object"] = np.tile([0.0] * 7 + [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], (T, 1))
            if outcomes is not None:
                g.attrs["success"], g.attrs["success_step"] = outcomes[i]
    return str(path)


def test_progress_labels():
    np.testing.assert_allclose(progress_labels(5, "demo"), [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(progress_labels(3, "random"), [0, 0, 0])
    np.testing.assert_allclose(progress_labels(6, "policy", 3), [0, 1 / 3, 2 / 3, 1, 1, 1])  # post-success = 1
    np.testing.assert_allclose(progress_labels(4, "policy", -1), [0, 0, 0, 0])  # failed rollout
    with pytest.raises(ValueError):
        progress_labels(3, "expert")


def test_dataset_labels_and_success(tmp_path):
    sources = [
        Source(make_file(tmp_path / "demo.hdf5", [3]), "demo"),
        Source(make_file(tmp_path / "rand.hdf5", [2]), "random"),
        Source(make_file(tmp_path / "pol.hdf5", [5, 4], [(True, 4), (False, -1)]), "policy"),
    ]
    ds = EncoderWMDataset(sources)
    assert len(ds) == 14
    assert ds.kind_counts() == {"demo": 3, "random": 2, "policy": 9}
    np.testing.assert_allclose(ds.progress.numpy(), [0, .5, 1, 0, 0, 0, .25, .5, .75, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(ds.success.numpy(), [1] * 3 + [0] * 2 + [1] * 5 + [0] * 4)
    assert ds.episode_success == [[True], [False], [True, False]]
    item = ds[12]
    assert item["kind"] == KIND_INDEX["policy"] and not item["success"] and not item["is_demo"]
    assert item["observation"]["agent_image"].shape == (3, 4, 4)
    assert ds[0]["is_demo"] and ds[0]["success"]


def test_policy_source_requires_outcomes(tmp_path):
    with pytest.raises(KeyError, match="success"):
        EncoderWMDataset([Source(make_file(tmp_path / "pol.hdf5", [3]), "policy")])
    with pytest.raises(ValueError):
        EncoderWMDataset([Source(make_file(tmp_path / "bad.hdf5", [3], [(True, -1)]), "policy")])
    with pytest.raises(ValueError):
        Source("x.hdf5", True)  # old boolean API


def test_sample_weights(tmp_path):
    sources = [
        Source(make_file(tmp_path / "demo.hdf5", [10]), "demo"),
        Source(make_file(tmp_path / "rand.hdf5", [30]), "random"),
        Source(make_file(tmp_path / "pol.hdf5", [60], [(False, -1)]), "policy"),
    ]
    ds = EncoderWMDataset(sources)
    w = ds.sample_weights(0.5, 0.3)
    mass = {k: w[ds.kind == i].sum().item() for k, i in KIND_INDEX.items()}
    assert mass == pytest.approx({"demo": 0.5, "random": 0.2, "policy": 0.3})
    # policy fraction redistributed when no policy data is present
    ds2 = EncoderWMDataset(sources[:2])
    w2 = ds2.sample_weights(0.5, 0.3)
    assert w2[ds2.is_demo].sum().item() == pytest.approx(0.5 / 0.7)
    with pytest.raises(ValueError):
        ds.sample_weights(0.8, 0.3)


def test_split_per_source(tmp_path):
    sources = [Source(make_file(tmp_path / "pol.hdf5", [2] * 10, [(False, -1)] * 10), "policy")]
    train, val = split_episodes(sources, 0.2, seed=0)
    assert len(val[sources[0].path]) == 2 and len(train[sources[0].path]) == 8
    ds = EncoderWMDataset(sources, val)
    assert len(ds) == 4
    assert torch.all(ds.progress == 0)
