# Local Linux / NVIDIA setup

The project runs from a plain Python virtual environment. These steps were verified on
Linux (Ubuntu, Python 3.12.3) with a laptop RTX 5060. CPU-only machines, macOS and Windows
have not been tested.

## 1. System prerequisites

- **NVIDIA driver with CUDA 13 support.** The pinned PyTorch (`torch==2.14.1`) uses CUDA 13.0
  wheels, so no system CUDA toolkit is needed, only the driver.
- **Python 3.12** with the `venv` module.
- **Native build tools, CMake 3 and EGL/OpenGL libraries.** `egl_probe` is compiled from source
  and fails with CMake 4. On Ubuntu:

  ```bash
  sudo apt-get install -y python3.12-venv git build-essential cmake \
      libegl1-mesa-dev libgl1-mesa-dev libglib2.0-0
  cmake --version   # must be 3.x
  ```

## 2. Create the virtual environment

From the repository root:

```bash
python3.12 -m venv robosuite-venv
source robosuite-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m pip check
```

`robosuite-venv/` is already in `.gitignore`. `requirements.txt` pins every package,
including robosuite 1.5.2, MuJoCo 3.3.1 and robomimic at a fixed Git commit. robomimic is
installed in editable mode, so pip clones it into `robosuite-venv/src/robomimic`.

Activate the environment (`source robosuite-venv/bin/activate`) in every new shell before
running anything below.

## 3. Rendering: on-screen vs. headless

- **On a machine with a display,** no extra setup is needed. Scripts that open a viewer
  (`scripts/random_action.py`, `test_policy.py` with `RENDER_GUI = True`) need a display.
- **On a headless machine or over SSH,** use EGL for off-screen rendering:

  ```bash
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  ```

  Don't set this when you want an on-screen viewer. `collect_random_data` sets these
  defaults itself unless you pass `--render`. Training scripts do not render.

## 4. Check the installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect True
python -m pytest -q                                                              # 18 tests
python scripts/random_action.py                                                  # needs a display
```

`random_action.py` opens a viewer with a Panda arm doing random actions in Lift. If it runs
and prints rewards, the simulator and renderer work. robosuite prints warnings about a
missing private macro file and optional packages (`robosuite_models`, mink IK). They are
harmless.

The Dockerfile and `nautilus/job.yaml` are only used for cluster training. They are
separate from this setup and not needed locally.

## 5. Prepare the Square demonstration dataset

Download the official proficient-human Square raw dataset at a fixed Hugging Face
repository revision:

```bash
mkdir -p datasets/square/ph
curl -fL --retry 3 \
  https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/74fa018461f479cd9fd15b924a16103012096203/v1.5/square/ph/demo_v15.hdf5 \
  -o datasets/square/ph/demo_v15.hdf5
sha256sum datasets/square/ph/demo_v15.hdf5
```

Expected SHA-256:
`3d9d6cadfa9395c5b368c029582dd73f1d4f699c1d82f78f0875e9bded61e596`.
Keep the raw file and render both camera views with next observations. Rendering is
off-screen, so export the EGL variables from step 3 first on a headless machine:

```bash
python -m robomimic.scripts.dataset_states_to_obs \
  --dataset datasets/square/ph/demo_v15.hdf5 \
  --output_name ../../square_image_v15.hdf5 \
  --done_mode 2 \
  --camera_names agentview robot0_eye_in_hand \
  --camera_height 84 --camera_width 84 --compress
```

These commands create files and should only be run with unused output paths.
The rendered file is `datasets/square_image_v15.hdf5` (200 demonstrations,
30,154 transitions). Source metadata records robosuite 1.5.0; rendering uses the
repo-pinned 1.5.2. This difference is recorded, not treated as exact simulator
version equivalence. `--done_mode 2` marks success or the final transition as done;
these demonstration labels are not separate termination/truncation labels.

Follow [random collection instructions](random-data-collection.md) to collect
additional transitions using this environment reference. For a short check:

```bash
python -m pytest -q
python -m scripts.datasets.collect_random_data \
  --reference-dataset datasets/square_image_v15.hdf5 \
  --output datasets/random/square_check.hdf5 \
  --episodes 2 --steps-per-episode 16 --seed 42
```

The [validation summary](random-data-validation.md) records the actual simulator
checks and initial random-data collection. Generated HDF5 files, local logs,
package inventories, detailed reports and previews stay outside Git.

## Next steps

- [Train the flow-matching policy](train-flow-policy.md)
- [Train the world-model encoder](train-wm-encoder.md)
