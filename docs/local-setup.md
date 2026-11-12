# Local Linux / NVIDIA setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate imitate-and-dream
python -m pip check
```

The environment uses Python 3.12, CUDA 12.8 PyTorch wheels (PyTorch 2.9.1 and
torchvision 0.24.1), robosuite 1.5.2, MuJoCo 3.3.1, and the same robomimic commit
as `requirements.txt`. It configures headless EGL rendering on activation.
CMake 3 is included because `egl-probe` 1.0.2 fails to build with CMake 4.
A working NVIDIA driver, EGL/OpenGL libraries, and native build tools (`make`,
C/C++ compiler) are required on the host. This configuration was verified on Linux
with an RTX 5070 Ti Laptop GPU; it has not been validated on CPU-only machines,
macOS, or Windows.

`requirements-local.txt` supplies this optional local environment. The original
`requirements.txt`, Dockerfile, and cluster job remain unchanged. Docker is not
required for the verified local workflow; the Docker image was not built or tested.

## Prepare the Square demonstration reference

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
Keep the raw file and render both camera views with next observations:

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
