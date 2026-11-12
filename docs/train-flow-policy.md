# Training the flow-matching policy

`scripts/imitation_policy/train_policy.py` trains `FlowMatchingPolicy`
(`models/imitation_policy/policy.py`) on robomimic image demonstrations. The policy maps
an observation to a chunk of 16 future actions.

## Prerequisites

- The environment from [local-setup.md](local-setup.md).
- An image dataset, e.g. `datasets/square_image_v15.hdf5` or `datasets/lift_image_v15.hdf5`
  (see [local-setup.md](local-setup.md) and `datasets/README.md`).

## Run

From the repository root:

```bash
python -m scripts.imitation_policy.train_policy \
    --dataset ./datasets/square_image_v15.hdf5 \
    --epochs 3000 --lr 1e-4 --warmup_steps 1000 --ema_decay 0.995 --save_every 1000
```

These are the settings used on the cluster (`nautilus/job.yaml`). The defaults
(`--dataset` is Lift, 100 epochs) are only meant for a quick run.

## What it does

- **Data:** `RoboMimicDataset` (`scripts/datasets/robomimic_dataset.py`). Each sample is
  the agent-view image, the wrist image, a 9-D proprio vector (eef pos, eef quat, gripper)
  and the next 16 actions. Chunks that run past the end of a demo are zero-padded and masked.
- **Model:** a separate small ViT for each camera, trained from scratch, plus a proprio MLP.
  Their tokens condition a transformer that predicts the flow velocity for the action chunk.
- **Objective:** flow matching. With t = 0 as noise and t = 1 as data, the target velocity is
  `actions - noise`. Random-shift augmentation (±4 px) is applied during training.
- **Optimization:** AdamW, linear warmup then cosine decay, grad-norm clip 1.0. An EMA copy of
  the weights is updated every step.
- **Validation:** the last `--val_frac` (5%) of samples, which is roughly the last demos.
  It runs on the EMA weights.

## Main arguments

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `./datasets/lift_image_v15.hdf5` | Demonstration HDF5 |
| `--epochs` | 100 | Passes over the training samples |
| `--batch_size` | 64 | |
| `--lr` / `--weight_decay` | 1e-4 / 1e-4 | AdamW |
| `--warmup_steps` | 500 | Linear LR warmup |
| `--ema_decay` | 0.995 | EMA of the weights, used for evaluation and rollouts |
| `--workers` | 0 | DataLoader workers |
| `--val_frac` / `--eval_every` | 0.05 / 5 | Held-out fraction, eval frequency in epochs |
| `--save_every` | 10 | Checkpoint frequency in epochs |
| `--out` | `checkpoints/imitation_policy/policy.pt` | Checkpoint path |
| `--wandb_mode` | `online` | `offline` or `disabled` to skip uploading |

## Outputs

- `--out`: the latest checkpoint, `{policy, ema, config, epoch}`.
- `<out>_best.pt`: the best checkpoint by `val/action_mse`.
- wandb project `flow-matching-policy`: metrics, a code snapshot, and both checkpoints as a
  model artifact.

## Metrics

- `train/loss`: flow-matching loss on augmented training batches.
- `val/flow_loss`: the same loss on held-out data, without augmentation.
- `val/action_mse`: MSE between one sampled chunk (10 Euler steps) and the demo chunk.
  It is noisy because the policy is multimodal. Use it for trends, not as a success measure.
- `val/action_mse_dim{i}`: the same error for each action dimension.

Task success is only measured by rollouts.

## Evaluate in simulation

Both scripts are configured with constants at the top of the file (`CHECKPOINT_NAME`,
`DATASET_PATH`, `NUM_ROLLOUTS`, `EXECUTION_HORIZON`, ...):

```bash
python -m scripts.imitation_policy.test_policy                 # success rate over NUM_ROLLOUTS episodes
./scripts/imitation_policy/test_oracle_success.sh 4            # pass@K oracle, 4 workers
```

Rollouts use the EMA weights and run 8 of every 16 predicted actions before replanning.
