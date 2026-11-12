import json
import os
import time

import cv2
import h5py
import numpy as np
import robosuite as suite
import robosuite.macros as macros
import torch

from models.imitation_policy.policy import FlowMatchingPolicy

# robosuite renders images upside-down by default (OpenGL convention), but the dataset images are upright
# (robomimic uses the OpenCV convention). This MUST be set before the environment is created.
macros.IMAGE_CONVENTION = "opencv"

CHECKPOINT_NAME = "checkpoints/nutAssemblyPolicy_3000_epoch.pt"
DATASET_PATH = "./datasets/square_image_v15.hdf5"  # env settings + reference frames; set to None to skip
DEBUG_DIR = "debug"
RENDER_GUI = True
NUM_ROLLOUTS = 30  # number of episodes to run
RECORD_VIDEO = False  # save an mp4 of every rollout (agent view | wrist view)
VIDEO_DIR = "videos"  # each run writes into VIDEO_DIR/<timestamp>/
VIDEO_SIZE = 256  # height and width of each camera view in the video
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_FLOW_STEPS = 10
NUM_EVAL_STEPS = 100
EXECUTION_HORIZON = 8   


def load_model(checkpoint_path):
    """Load the EMA Flow Matching Policy from checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    # Reconstruct architecture using saved training configuration
    config = checkpoint["config"]

    model = FlowMatchingPolicy(**config)

    # Load EMA weights (recommended for evaluation)
    model.load_state_dict(checkpoint["ema"])

    model.to(DEVICE)
    model.eval()

    print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")
    print(f"Model configuration: {config}")
    print("EMA model loaded successfully!")

    return model


def create_environment():
    """Create the Lift environment with the same settings the dataset was recorded with."""

    env_name = "Lift"
    kwargs = dict(
        robots="Panda",
        use_object_obs=True,
        reward_shaping=False,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=84,
        camera_widths=84,
        camera_depths=False,
        control_freq=20,
    )

    if DATASET_PATH and os.path.exists(DATASET_PATH):
        # robomimic stores the exact env settings (task, controller, cameras, ...) inside the hdf5
        with h5py.File(DATASET_PATH, "r") as f:
            meta = json.loads(f["data"].attrs["env_args"])
        env_name, kwargs = meta["env_name"], dict(meta["env_kwargs"])

        dataset_version = meta.get("env_version")
        print(f"robosuite: dataset recorded with {dataset_version}, installed {suite.__version__}")
        if dataset_version and dataset_version != suite.__version__:
            print("WARNING: robosuite versions differ; camera poses / rendering can differ between versions")
    else:
        print("WARNING: dataset not found, using hard-coded environment settings")

    kwargs.update(
        has_renderer=RENDER_GUI,
        has_offscreen_renderer=True,
        use_camera_obs=True,
    )
    return suite.make(env_name, **kwargs)


def save_comparison(obs):
    """Save dataset frame 0 (left) next to the environment's reset frame (right): agent view on top, wrist below.

    If the two do not look alike (flipped, different viewpoint, ...) the policy will not work.
    """

    if not (DATASET_PATH and os.path.exists(DATASET_PATH)):
        return

    with h5py.File(DATASET_PATH, "r") as f:
        demo_obs = f["data"][sorted(f["data"].keys())[0]]["obs"]
        reference = {
            "agentview_image": demo_obs["agentview_image"][0],
            "robot0_eye_in_hand_image": demo_obs["robot0_eye_in_hand_image"][0],
        }

    rows = []
    for key, ref_image in reference.items():
        env_image = np.ascontiguousarray(obs[key])
        diff = np.abs(ref_image.astype(float) - env_image.astype(float)).mean()
        print(f"{key}: mean abs pixel difference to dataset frame 0 = {diff:.1f} "
              "(about 5-10 when the views match; the cube position differs; >30 means flipped / different view)")
        rows.append(np.concatenate([ref_image, env_image], axis=1))

    grid = cv2.resize(np.concatenate(rows, axis=0), None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, "dataset_vs_env.png")
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved comparison (left: dataset, right: environment) to {path}")


class VideoRecorder:
    """Writes RGB frames to an mp4 file.

    Uses imageio (H.264, plays everywhere) when `pip install imageio imageio-ffmpeg` is available,
    otherwise falls back to OpenCV's mp4v codec (plays in VLC / most desktop players, not in browsers).
    """

    def __init__(self, path, fps, frame_shape):
        try:
            import imageio

            writer = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p",
                                        quality=8, macro_block_size=1)
            self._write, self._close = writer.append_data, writer.close
        except Exception as e:  # imageio / imageio-ffmpeg missing
            print(f"imageio unavailable ({type(e).__name__}); recording with OpenCV mp4v instead")
            height, width = frame_shape[:2]
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            self._write = lambda frame: writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self._close = writer.release

    def write(self, frame):
        self._write(frame)

    def close(self):
        self._close()


def render_frame(env, label):
    """Agent view and wrist view side by side (upright, VIDEO_SIZE px each) with a text label."""

    # sim.render always returns OpenGL (upside-down) images, independent of IMAGE_CONVENTION -> flip
    views = [
        env.sim.render(camera_name=camera, width=VIDEO_SIZE, height=VIDEO_SIZE)[::-1]
        for camera in ("agentview", "robot0_eye_in_hand")
    ]
    frame = np.ascontiguousarray(np.concatenate(views, axis=1))

    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):  # outlined text, readable on any background
        cv2.putText(frame, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)

    return frame


def preprocess_image(image):
    """Convert HWC image to normalized BCHW tensor."""

    image = torch.as_tensor(
        image.copy(),
        dtype=torch.float32,
        device=DEVICE
    )

    image = image.permute(2, 0, 1) / 255.0

    return image.unsqueeze(0)


def preprocess_observation(obs):
    """Prepare camera and proprioception observations."""

    agent_image = preprocess_image(obs["agentview_image"])
    wrist_image = preprocess_image(
        obs["robot0_eye_in_hand_image"]
    )

    proprio = np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["robot0_gripper_qpos"],
    ])

    proprio = torch.as_tensor(
        proprio,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    return {
        "agent_image": agent_image,
        "wrist_image": wrist_image,
        "proprio": proprio,
    }


def predict_action_chunk(model, obs):
    """Generate an action chunk using flow matching."""

    inputs = preprocess_observation(obs)

    with torch.no_grad():
        action_chunk = model.predict(
            inputs,
            num_steps=NUM_FLOW_STEPS
        )

    return action_chunk.squeeze(0).cpu().numpy()


def evaluate_policy(env, model, rollout_idx=0, video_dir=None):
    """Execute action chunks using receding-horizon control. Returns True if the rollout succeeded."""

    obs = env.reset()
    if rollout_idx == 0:
        save_comparison(obs)

    recorder, video_path = None, None
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"rollout_{rollout_idx:03d}.mp4")
        first_frame = render_frame(env, f"rollout {rollout_idx}  t=0")
        recorder = VideoRecorder(video_path, int(env.control_freq), first_frame.shape)
        recorder.write(first_frame)

    reward, done, success = 0.0, False, False
    try:
        for step in range(NUM_EVAL_STEPS):

            action_chunk = predict_action_chunk(model, obs)

            # Execute the first EXECUTION_HORIZON actions of the predicted chunk
            for action in action_chunk[:EXECUTION_HORIZON]:

                obs, reward, done, info = env.step(action)

                if RENDER_GUI:
                    env.render()

                # robosuite's `done` only fires at the horizon (1000 steps); success must be checked explicitly
                success = env._check_success()

                if recorder is not None:
                    label = f"rollout {rollout_idx}  t={env.timestep}" + ("  SUCCESS" if success else "")
                    recorder.write(render_frame(env, label))

                if done or success:
                    break

            if done or success:
                break

            if step % 20 == 0:
                print(f"Step: {step} | Reward: {reward:.4f}")
    finally:
        if recorder is not None:
            recorder.close()

    print(f"Rollout {rollout_idx + 1}: {'SUCCESS' if success else 'no success'} after {env.timestep} env steps")

    if video_path is not None:  # mark the outcome in the file name
        final_path = video_path.replace(".mp4", "_success.mp4" if success else "_fail.mp4")
        os.replace(video_path, final_path)
        print(f"Saved video to {final_path}")

    return success


def main():

    model = load_model(CHECKPOINT_NAME)

    env = create_environment()

    video_dir = os.path.join(VIDEO_DIR, time.strftime("%Y%m%d_%H%M%S")) if RECORD_VIDEO else None

    results = []
    try:
        for rollout_idx in range(NUM_ROLLOUTS):
            print(f"\n=== Rollout {rollout_idx + 1}/{NUM_ROLLOUTS} ===")
            results.append(evaluate_policy(env, model, rollout_idx, video_dir))

    finally:
        env.close()

    print(f"\nSuccess rate: {sum(results)}/{len(results)}")
    if video_dir is not None:
        print(f"Videos saved in {video_dir}")


if __name__ == "__main__":
    main()