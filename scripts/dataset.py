import h5py
import numpy as np
import torch

from torch.utils.data import Dataset


class RoboMimicDataset(Dataset):
    def __init__(
        self,
        hdf5_path,
        action_horizon=16,
        agent_camera_key="agentview_image",
        wrist_camera_key="robot0_eye_in_hand_image",
        proprio_keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
    ):
        self.hdf5_path = hdf5_path
        self.action_horizon = action_horizon
        self.agent_camera_key = agent_camera_key
        self.wrist_camera_key = wrist_camera_key
        self.proprio_keys = list(proprio_keys)

        if action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")

        # Read metadata without loading the full dataset into RAM.
        with h5py.File(self.hdf5_path, "r") as f:
            data = f["data"]

            self.demos = sorted(data.keys())
            self.lengths = []

            for demo_name in self.demos:
                demo = data[demo_name]
                obs = demo["obs"]

                # Check that all required observations exist.
                required_keys = [
                    self.agent_camera_key,
                    self.wrist_camera_key,
                    *self.proprio_keys,
                ]

                missing = [
                    key for key in required_keys
                    if key not in obs
                ]

                if missing:
                    raise KeyError(
                        f"{demo_name} is missing observation keys: "
                        f"{missing}"
                    )

                # Validate that observations and actions align.
                T = len(demo["actions"])

                for key in required_keys:
                    if len(obs[key]) != T:
                        raise ValueError(
                            f"Length mismatch in {demo_name}: "
                            f"{key} has {len(obs[key])} frames, "
                            f"but actions has {T}."
                        )

                self.lengths.append(T)

        # Map global sample indices to demonstrations.
        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_samples = int(self.cumulative_lengths[-1])

        # Each worker opens its own HDF5 handle on demand.
        self.file = None

    def __len__(self):
        return self.total_samples

    def __getstate__(self):
        # Avoid serializing an open HDF5 handle to DataLoader workers.
        state = self.__dict__.copy()
        state["file"] = None
        return state

    def _get_file(self):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r")
        return self.file

    @staticmethod
    def _process_image(image):
        """
        Input:  H x W x C, usually uint8
        Output: C x H x W, float32 in [0, 1]
        """
        image = np.asarray(image)

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, 3), "
                f"got {image.shape}"
            )

        image = np.transpose(image, (2, 0, 1)).copy()

        return torch.from_numpy(image).float() / 255.0

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.total_samples

        if idx < 0 or idx >= self.total_samples:
            raise IndexError(idx)

        # Find demonstration and local timestep.
        demo_idx = np.searchsorted(
            self.cumulative_lengths,
            idx,
            side="right",
        )

        previous_end = (
            self.cumulative_lengths[demo_idx - 1]
            if demo_idx > 0
            else 0
        )

        timestep = int(idx - previous_end)
        demo_name = self.demos[demo_idx]

        f = self._get_file()
        demo = f["data"][demo_name]
        obs = demo["obs"]

        # --------------------------------------------------
        # 1. External camera image
        # --------------------------------------------------
        agent_image = self._process_image(
            obs[self.agent_camera_key][timestep]
        )

        # --------------------------------------------------
        # 2. Wrist-mounted camera image
        # --------------------------------------------------
        wrist_image = self._process_image(
            obs[self.wrist_camera_key][timestep]
        )

        # --------------------------------------------------
        # 3. Proprioceptive state
        # --------------------------------------------------
        proprio = np.concatenate([
            np.asarray(obs[key][timestep]).reshape(-1)
            for key in self.proprio_keys
        ]).astype(np.float32)

        proprio = torch.from_numpy(proprio)

        # --------------------------------------------------
        # 4. Chunk of future actions
        # --------------------------------------------------
        actions = demo["actions"]
        T = len(actions)

        end = min(timestep + self.action_horizon, T)

        # Do not allow action chunks to cross episode boundaries.
        action_chunk = np.asarray(
            actions[timestep:end],
            dtype=np.float32,
        )

        valid_length = len(action_chunk)

        # Pad incomplete chunks with zeros.
        # The mask identifies which actions are valid.
        if valid_length < self.action_horizon:
            pad_length = self.action_horizon - valid_length

            padding = np.zeros(
                (pad_length, *action_chunk.shape[1:]),
                dtype=np.float32,
            )

            action_chunk = np.concatenate(
                [action_chunk, padding],
                axis=0,
            )

        action_chunk = torch.from_numpy(action_chunk)

        action_mask = torch.zeros(
            self.action_horizon,
            dtype=torch.bool,
        )
        action_mask[:valid_length] = True

        return {
            "observation": {
                "agent_image": agent_image,
                "wrist_image": wrist_image,
                "proprio": proprio,
            },
            "action_chunk": action_chunk,
            "action_mask": action_mask,
        }

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None