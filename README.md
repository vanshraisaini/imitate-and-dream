# Robot Learning with Imitation Learning and World Models

This project explores modern robot learning for manipulation using
imitation learning and world models. It uses robotic demonstration datasets to learn generative
policies from visual observations and proprioceptive states, with the
goal of predicting chunks of future actions.

## Setup

### Local Conda environment (Linux / NVIDIA)

```bash
conda env create -f environment.yml
conda activate imitate-and-dream
```

This configures headless EGL rendering and installs the local dependency set.
Docker is optional for local development. Dataset download, image conversion,
and verification instructions are in [local setup instructions](docs/local-setup.md).
The Square dataset preparation commands produce `datasets/square_image_v15.hdf5`.

### Original virtualenv setup

#### 1. Create a virtual environment

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install dependencies

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Test the Setup

Run the random actions script to verify that the environment and
simulator are working correctly:

``` bash
python scripts/random_action.py
```

If the script runs successfully, your environment is ready for further
development and experimentation.

## Random-action dataset collection

```bash
conda activate imitate-and-dream
python -m scripts.collect_random_data \
  --reference-dataset datasets/square_image_v15.hdf5 \
  --output datasets/random/square_random.hdf5 \
  --episodes 10 --steps-per-episode 400 --seed 42
```

The collector saves both camera views and proprioception before and after each
bounded random action. See [collection instructions](docs/random-data-collection.md)
for other environments, the HDF5 schema, and output protection.
