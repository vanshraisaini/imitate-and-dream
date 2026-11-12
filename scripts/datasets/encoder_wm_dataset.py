"""
Frame dataset for training the world-model encoder on demonstrations + random-action data.

Each item is one frame:
    observation:  agent_image (3, H, W) uint8, wrist_image (3, H, W) uint8, proprio (P,) float32
    aux_target:   (21,) privileged state, layout in models/world_model/aux_head.py
    progress:     () t / (T - 1) for demonstrations, 0 for random-action episodes
    is_demo:      () bool

Images are returned as uint8 (the trainer converts on GPU); all low-dim data is loaded into
memory at construction, so __getitem__ only reads two images from disk.
"""
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from models.world_model.aux_head import TARGET_DIM, build_aux_targets
from models.world_model.progress_head import progress_labels

AGENT_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


@dataclass(frozen=True)
class Source:
    path: str
    is_demo: bool


def _sorted_episodes(f):
    # demo_0, demo_1, ..., demo_10 (numeric, not lexicographic) so splits are reproducible.
    return sorted(f["data"].keys(), key=lambda n: (len(n), n))


def split_episodes(sources, val_frac, seed=0):
    """Hold out val_frac of the episodes of EACH source. Returns (train, val): {path: [names]}."""
    rng = np.random.default_rng(seed)
    train, val = {}, {}
    for src in sources:
        with h5py.File(src.path, "r") as f:
            names = _sorted_episodes(f)
        perm = rng.permutation(len(names))
        n_val = int(round(len(names) * val_frac))
        val[src.path] = sorted((names[i] for i in perm[:n_val]), key=lambda n: (len(n), n))
        train[src.path] = sorted((names[i] for i in perm[n_val:]), key=lambda n: (len(n), n))
    return train, val


class EncoderWMDataset(Dataset):
    def __init__(self, sources, episodes=None):
        """
        sources:  list of Source(path, is_demo)
        episodes: optional {path: [episode names]} to restrict each source (e.g. from split_episodes)
        """
        self.sources = list(sources)
        self.files = [None] * len(self.sources)  # opened lazily, once per DataLoader worker

        src_idx, ep_names, steps = [], [], []
        proprio, aux, progress, is_demo = [], [], [], []
        self.episode_names = []  # per source
        for s, src in enumerate(self.sources):
            with h5py.File(src.path, "r") as f:
                names = _sorted_episodes(f) if episodes is None else list(episodes.get(src.path, []))
                self.episode_names.append(names)
                for e, name in enumerate(names):
                    obs = f["data"][name]["obs"]
                    T = len(obs[AGENT_KEY])
                    for k in (WRIST_KEY, *PROPRIO_KEYS, "object"):
                        if k not in obs:
                            raise KeyError(f"{src.path}:{name} is missing {k!r}")
                        if len(obs[k]) != T:
                            raise ValueError(f"{src.path}:{name}: {k} has {len(obs[k])} frames, expected {T}")
                    proprio.append(np.concatenate(
                        [np.asarray(obs[k][:], dtype=np.float32).reshape(T, -1) for k in PROPRIO_KEYS], axis=1))
                    aux.append(build_aux_targets(obs["object"][:], obs["robot0_eef_pos"][:],
                                                 obs["robot0_eef_quat"][:]))
                    progress.append(progress_labels(T, src.is_demo))
                    is_demo.append(np.full(T, src.is_demo))
                    src_idx.append(np.full(T, s))
                    ep_names.append(np.full(T, e))
                    steps.append(np.arange(T))

        if not steps:
            raise ValueError("no frames selected")
        self.src_idx = np.concatenate(src_idx)
        self.ep_idx = np.concatenate(ep_names)
        self.t = np.concatenate(steps)
        self.proprio = torch.from_numpy(np.concatenate(proprio))
        self.aux_targets = torch.from_numpy(np.concatenate(aux))
        self.progress = torch.from_numpy(np.concatenate(progress))
        self.is_demo = torch.from_numpy(np.concatenate(is_demo))
        assert self.aux_targets.shape[1] == TARGET_DIM

    def __len__(self):
        return len(self.t)

    def __getstate__(self):
        # Never send open HDF5 handles to DataLoader workers.
        state = self.__dict__.copy()
        state["files"] = [None] * len(self.sources)
        return state

    def _file(self, s):
        if self.files[s] is None:
            self.files[s] = h5py.File(self.sources[s].path, "r")
        return self.files[s]

    @staticmethod
    def _image(x):
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[-1] != 3 or x.dtype != np.uint8:
            raise ValueError(f"expected (H, W, 3) uint8 image, got {x.dtype} {x.shape}")
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def __getitem__(self, idx):
        s, e, t = int(self.src_idx[idx]), int(self.ep_idx[idx]), int(self.t[idx])
        obs = self._file(s)["data"][self.episode_names[s][e]]["obs"]
        return {
            "observation": {
                "agent_image": self._image(obs[AGENT_KEY][t]),
                "wrist_image": self._image(obs[WRIST_KEY][t]),
                "proprio": self.proprio[idx],
            },
            "aux_target": self.aux_targets[idx],
            "progress": self.progress[idx],
            "is_demo": self.is_demo[idx],
        }

    def sample_weights(self, demo_frac):
        """
        Per-frame weights for a WeightedRandomSampler so that a fraction demo_frac of the
        drawn frames are demonstrations, regardless of how many frames each group has.
        """
        if not 0.0 < demo_frac < 1.0:
            raise ValueError("demo_frac must be in (0, 1)")
        n_demo = int(self.is_demo.sum())
        n_rand = len(self) - n_demo
        if n_demo == 0 or n_rand == 0:
            return torch.ones(len(self), dtype=torch.double)
        return torch.where(self.is_demo, demo_frac / n_demo, (1.0 - demo_frac) / n_rand).double()

    def close(self):
        for f in self.files:
            if f is not None:
                f.close()
        self.files = [None] * len(self.sources)
