"""
Frame dataset for training the world-model encoder on demonstrations, random-action data and
policy rollouts together.

Each item is one frame:
    observation:  agent_image (3, H, W) uint8, wrist_image (3, H, W) uint8, proprio (P,) float32
    aux_target:   (21,) privileged state, layout in models/world_model/aux_head.py
    progress:     () progress label, see models/world_model/progress_head.py:progress_labels
                    demo t / (T - 1) | successful rollout min(t / t_success, 1) | failed rollout 0 | random 0
    kind:         () int64 index into EPISODE_KINDS ("demo", "random", "policy")
    is_demo:      () bool, kind == "demo"
    success:      () bool, the episode's outcome (demos True, random False, rollouts from the file)

Source files:
    demo    robomimic demonstrations (e.g. datasets/square_image_v15.hdf5)
    random  scripts/datasets/collect_random_data.py output
    policy  scripts/datasets/collect_policy_data.py output (needs the per-episode success /
            success_step attributes it writes)

Images are returned as uint8 (the trainer converts on GPU); all low-dim data is loaded into
memory at construction, so __getitem__ only reads two images from disk.
"""
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from models.world_model.aux_head import TARGET_DIM, build_aux_targets
from models.world_model.progress_head import EPISODE_KINDS, progress_labels

AGENT_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
KIND_INDEX = {k: i for i, k in enumerate(EPISODE_KINDS)}


@dataclass(frozen=True)
class Source:
    path: str
    kind: str  # "demo", "random" or "policy"

    def __post_init__(self):
        if self.kind not in KIND_INDEX:
            raise ValueError(f"{self.path}: unknown kind {self.kind!r}, expected one of {EPISODE_KINDS}")


def _sorted_episodes(f):
    # demo_0, demo_1, ..., demo_10 (numeric, not lexicographic) so splits are reproducible.
    return sorted(f["data"].keys(), key=lambda n: (len(n), n))


def _episode_outcome(src, group):
    """(success, success_step) for one episode. success_step is -1 if the task was never completed."""
    if src.kind == "demo":
        return True, -1
    if src.kind == "random":
        return False, -1
    if "success" not in group.attrs or "success_step" not in group.attrs:
        raise KeyError(f"{src.path}:{group.name} has no success / success_step attributes; "
                       "policy sources must come from collect_policy_data.py")
    success, step = bool(group.attrs["success"]), int(group.attrs["success_step"])
    if success != (step >= 0):
        raise ValueError(f"{src.path}:{group.name}: success={success} but success_step={step}")
    return success, step


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
        sources:  list of Source(path, kind)
        episodes: optional {path: [episode names]} to restrict each source (e.g. from split_episodes)
        """
        self.sources = list(sources)
        self.files = [None] * len(self.sources)  # opened lazily, once per DataLoader worker

        src_idx, ep_names, steps = [], [], []
        proprio, aux, progress, kind, success = [], [], [], [], []
        self.episode_names = []  # per source
        self.episode_success = []  # per source, list of bools
        for s, src in enumerate(self.sources):
            with h5py.File(src.path, "r") as f:
                names = _sorted_episodes(f) if episodes is None else list(episodes.get(src.path, []))
                self.episode_names.append(names)
                self.episode_success.append([])
                for e, name in enumerate(names):
                    group = f["data"][name]
                    obs = group["obs"]
                    T = len(obs[AGENT_KEY])
                    for k in (WRIST_KEY, *PROPRIO_KEYS, "object"):
                        if k not in obs:
                            raise KeyError(f"{src.path}:{name} is missing {k!r}")
                        if len(obs[k]) != T:
                            raise ValueError(f"{src.path}:{name}: {k} has {len(obs[k])} frames, expected {T}")
                    ep_success, success_step = _episode_outcome(src, group)
                    self.episode_success[s].append(ep_success)
                    proprio.append(np.concatenate(
                        [np.asarray(obs[k][:], dtype=np.float32).reshape(T, -1) for k in PROPRIO_KEYS], axis=1))
                    aux.append(build_aux_targets(obs["object"][:], obs["robot0_eef_pos"][:],
                                                 obs["robot0_eef_quat"][:]))
                    progress.append(progress_labels(T, src.kind, success_step))
                    kind.append(np.full(T, KIND_INDEX[src.kind]))
                    success.append(np.full(T, ep_success))
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
        self.kind = torch.from_numpy(np.concatenate(kind).astype(np.int64))
        self.is_demo = self.kind == KIND_INDEX["demo"]
        self.success = torch.from_numpy(np.concatenate(success))
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
            "kind": self.kind[idx],
            "is_demo": self.is_demo[idx],
            "success": self.success[idx],
        }

    def kind_counts(self):
        """{kind: number of frames} for the kinds present."""
        return {k: int((self.kind == i).sum()) for k, i in KIND_INDEX.items() if (self.kind == i).any()}

    def sample_weights(self, demo_frac, policy_frac=0.0):
        """
        Per-frame weights for a WeightedRandomSampler so that a fraction demo_frac of the drawn
        frames are demonstrations, policy_frac are policy rollouts and the rest random-action
        frames, regardless of how many frames each group has. Fractions of kinds that are absent
        are redistributed proportionally over the kinds that are present.
        """
        fracs = {"demo": demo_frac, "policy": policy_frac, "random": 1.0 - demo_frac - policy_frac}
        if any(f < 0 or f > 1 for f in fracs.values()):
            raise ValueError(f"fractions must be in [0, 1] and sum to at most 1, got {fracs}")
        counts = self.kind_counts()
        total = sum(fracs[k] for k in counts)
        if total <= 0:
            raise ValueError(f"all present kinds {list(counts)} have sampling fraction 0")
        w = torch.zeros(len(self), dtype=torch.double)
        for k, n in counts.items():
            w[self.kind == KIND_INDEX[k]] = fracs[k] / total / n
        return w

    def close(self):
        for f in self.files:
            if f is not None:
                f.close()
        self.files = [None] * len(self.sources)
