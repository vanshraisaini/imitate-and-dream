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