Kindly refer to [this](https://github.com/ARISE-Initiative/robomimic/blob/master/docs/datasets/robomimic_v0.1.md?utm_source=chatgpt.com) to setup the prerequisites to install datasets.


Example command to download raw dataset:

```
python robomimic/scripts/download_datasets.py --tasks lift --dataset_types ph --hdf5_types raw
```

Example command to render images for the dataset:

```
python robomimic/scripts/dataset_states_to_obs.py \
  --dataset datasets/lift/ph/demo_v141.hdf5 \
  --output_name image_v141.hdf5 \
  --done_mode 2 \
  --camera_names agentview robot0_eye_in_hand \
  --camera_height 84 \
  --camera_width 84
```

## Prepared local Square dataset

The local setup uses the proficient-human Square dataset from the official
robomimic v1.5 release:

- Original raw file: `datasets/square/ph/demo_v15.hdf5`
- Rendered observations: `datasets/square_image_v15.hdf5`
- Both `agentview` and `robot0_eye_in_hand` cameras at 84 x 84 RGB
- Both `obs` and `next_obs` retained, with gzip compression

See [setup and regeneration instructions](../docs/local-setup.md).
Dataset files are ignored by Git.

## Random-action data

Use `python -m scripts.datasets.collect_random_data` to collect transitions from the same
environment as a demonstration file, or from a JSON environment config. See
[random collection instructions](../docs/random-data-collection.md). Keep random
data in a separate file from expert demonstrations: it is intended as additional
world-model training data, not as imitation-learning targets.


An initial Square random-action batch was collected locally at
`datasets/random/square_random_10x400_seed42.hdf5`: 10 episodes, 4,000 transitions,
62.77 MiB. See the [validation summary](../docs/random-data-validation.md)
for validation, checksums, regeneration commands and limitations.
