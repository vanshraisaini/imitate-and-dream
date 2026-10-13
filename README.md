# Robot Learning with Imitation Learning

This project explores modern robot learning for manipulation using
imitation learning and world models. It uses robotic demonstration datasets to learn generative
policies from visual observations and proprioceptive states, with the
goal of predicting chunks of future actions.

## Setup

### 1. Create a virtual environment

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

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
