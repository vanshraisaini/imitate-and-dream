# Random-action data collection (issue #1)

The collector records `(observation, action, next_observation)` transitions from
robosuite. Both camera images and proprioception are saved before and after each
action. Actions are sampled independently and uniformly within the environment's
per-dimension `action_spec`. No learned policy or training process is involved.

## Collect from the demonstration environment

From the repository root:

```bash
conda activate imitate-and-dream
python -m scripts.collect_random_data \
  --reference-dataset datasets/square_image_v15.hdf5 \
  --output datasets/random/square_random.hdf5 \
  --episodes 10 --steps-per-episode 400 --seed 42
```

This retains the reference task, robot, controller, control frequency and physics
settings. It enables offscreen cameras, disables depth, and seeds the environment.
Image resolution is inferred from the reference images when available, otherwise
from scalar camera dimensions in its environment metadata (84 x 84 if absent).
`--image-height` and `--image-width` explicitly override this. The original
reference metadata, absolute path and SHA-256 are recorded in the output.
Simulator version differences are printed and retained in the provenance.

Use the processed demonstration file to match its rendered observations. The raw
demonstration HDF5 also works when it contains `data.attrs['env_args']`.

## Select a different environment

```bash
python -m scripts.collect_random_data \
  --env-config configs/collection/lift.json \
  --output datasets/random/lift_random.hdf5 \
  --episodes 10 --steps-per-episode 400 --seed 42
```

The JSON uses the same `env_name` / `env_kwargs` structure as robomimic metadata.
Set task-specific robots, controllers, horizons and control frequency in
`env_kwargs`. Defaults resolved internally by robosuite are version-dependent;
use a reference dataset or explicit controller configuration when exact matching
is needed. Output metadata stores the supplied kwargs and installed simulator
version. The example uses Panda's installed default controller.

`--camera-names AGENT WRIST` selects exactly two cameras, defaulting to
`agentview robot0_eye_in_hand`. `--proprio-keys KEY [KEY ...]` selects finite
vector observations, defaulting to `robot0_eef_pos`, `robot0_eef_quat`, and
`robot0_gripper_qpos` (9 values for Panda). These options support robots with
different observation names. Missing observations fail explicitly.

Headless collection defaults to EGL on Linux. Existing `MUJOCO_GL` and
`PYOPENGL_PLATFORM` settings are respected. Use `--render` from a display-enabled
terminal to also show an interactive viewer.

## HDF5 layout

```text
root attrs: schema_version, collection_complete, collection_config (JSON)
data attrs: env_args (JSON), total
data/demo_0 attrs: num_samples
data/demo_0/
  obs/agentview_image          uint8 [T, H, W, 3], upright RGB
  obs/robot0_eye_in_hand_image uint8 [T, H, W, 3], upright RGB
  obs/robot0_eef_pos           [T, 3]
  obs/robot0_eef_quat          [T, 4]
  obs/robot0_gripper_qpos      [T, 2]
  next_obs/                   same observation keys and shapes
  actions                     float64 [T, action_dim]
  rewards                     float64 [T]
  env_dones                   bool [T]
  collection_truncated        bool [T]
  dones                       bool [T]
data/demo_1/ ...
```

Images and proprioception use lossless gzip compression. Arrays preserve native
observation dtypes; image values remain raw uint8 rather than normalized tensors.
The current imitation loader can read the default camera/proprioception layout.
It still returns action chunks; it does not expose `next_obs` for world-model
training. Custom keys must also be supplied to that loader.

`env_dones` is exactly the simulator's returned `done`. `collection_truncated`
marks the final transition when the collector reaches its step limit while
`env_done` is false. `dones` is their union, marking an episode boundary. In
robosuite a simulator `done` may itself represent a time horizon: these fields
must not be interpreted as absorbing-state labels or discount masks without
checking the environment settings. Success does not stop collection. The final
post-action observation is saved before any reset.

`collection_config` records seed, sampling method, bounds, camera/proprioception
keys, image convention, requested counts, source provenance, library versions and
the collector source SHA-256. The file contains observations, not saved MuJoCo
states/model XML, so it is not an input to robomimic's state-to-image converter.

## Output protection and memory

Neither existing output files nor existing `<output>.part` files are overwritten.
Collection buffers one episode at a time, so memory scales with episode length
and image resolution rather than the requested number of episodes. Large runs
should use bounded episode lengths; there is no automatic sharding or resume.

Episodes are flushed to `<output>.part`; only after the entire collection closes
successfully is the final path published atomically using a same-directory hard
link. The destination filesystem must support hard links. Exceptions and
interruptions leave `.part` for inspection, without publishing a finished file.
An interruption during writing can leave the current episode incomplete; inspect
the file and committed sample counts before attempting any manual recovery.
Choose a new output path to retry. The environment is closed on collection errors.

## Verification

```bash
python -m pytest -q
```

Focused tests cover mutable observation buffers, action preservation and bounds,
seeded action sampling, simulator termination versus collection limits, invalid
inputs, overwrite refusal, and interrupted-file behavior. Actual simulator smoke
collections and the verification report are documented in
[the validation summary](random-data-validation.md).

Independent uniform actions are a baseline exploration policy. Producing this
dataset does not establish improved world-model coverage or prediction quality;
those require subsequent distribution analysis and model evaluation.
