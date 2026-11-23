# Training the world-model encoder

`scripts/world_model/train_wm_encoder.py` trains the encoder that maps an observation to
a 256-D latent `z`. Later the latent dynamics model rolls `z` forward and a progress head
scores the result. The encoder is trained with two supervised heads only. There is no
reconstruction decoder and no EMA target.

## Prerequisites

- The environment from [local-setup.md](local-setup.md).
- Demonstrations: `datasets/square_image_v15.hdf5`.
- Random-action episodes in the same environment, which must include the `object` key:

  ```bash
  python -m scripts.datasets.collect_random_data \
      --reference-dataset datasets/square_image_v15.hdf5 \
      --output datasets/random/square_random.hdf5 \
      --episodes 200 --steps-per-episode 200 --seed 42
  ```

  Random actions almost never move the nut, so each episode adds only one nut pose. Many
  short episodes (e.g. 1000 × 40) give more pose diversity than a few long ones.
- Optional: rollouts of the frozen policy with success labels. They teach the progress head
  what a near-expert *failure* looks like, which demos and random actions cannot:

  ```bash
  python -m scripts.datasets.collect_policy_data \
      --reference-dataset datasets/square_image_v15.hdf5 \
      --output datasets/policy/square_policy_s0.hdf5 --episodes 100 --start-seed 0
  ```
- DINOv2 is downloaded from `torch.hub` on first use.

## Run

From the repository root:

```bash
python -m scripts.world_model.train_wm_encoder \
    --demo_datasets datasets/square_image_v15.hdf5 \
    --random_datasets datasets/random/square_random.hdf5 \
    --policy_datasets datasets/policy/square_policy_s*.hdf5 \
    --epochs 30 --amp
```

`--demo_datasets`, `--random_datasets` and `--policy_datasets` each accept several files;
`--policy_datasets` can be left out. One epoch takes about a
minute on a laptop RTX 5060.

## What it does

- **Data:** `EncoderWMDataset` (`scripts/datasets/encoder_wm_dataset.py`). One item is one
  frame: both camera images (uint8), the 9-D proprio vector, a 21-D aux target, a progress
  label, the episode kind (`demo` / `random` / `policy`), an `is_demo` flag and the episode's
  outcome (`success`). Low-dim data is loaded into memory; images are read from disk.
- **Labels:**
  - Aux target: nut world pose, nut pose in the gripper frame, gripper pose. It is built by
    `build_aux_targets` in `models/world_model/aux_head.py`, which recomputes the relative
    pose because robosuite's `object[0:7]` is one step stale.
  - Progress (`progress_labels` in `models/world_model/progress_head.py`):

    | Episode | Label |
    |---|---|
    | demo | `t / (T - 1)` |
    | successful policy rollout | `min(t / t_success, 1)` |
    | failed policy rollout | 0 |
    | random actions | 0 |

    Early frames of successful and failed rollouts look alike, so the squared loss pulls the
    head towards P(success | state) × progress, a value-like score.
- **Split:** `--val_frac` of the episodes of *each* source is held out. The split is stored
  in the checkpoint.
- **Sampling:** a weighted sampler makes `--demo_frac` of every batch demo frames,
  `--policy_frac` policy frames and the rest random-action frames. The share of a missing
  source is redistributed over the others.
- **Model:**
  - Frozen DINOv2 ViT-S/14 turns each 84×84 image into 36 patch tokens.
  - A fusion transformer combines the patch tokens from both cameras, a proprio token and a
    CLS token. `z` is the CLS output after a LayerNorm.
  - Two MLP heads read `z`: `AuxStateHead` (21-D state) and `ProgressHead` (g in [0, 1]).
- **Loss:** `lambda_aux * L_aux + lambda_prog * L_progress`. Both losses update the fusion
  transformer and the proprio MLP; DINOv2 gets no gradient. Random-shift augmentation
  (±4 px, each camera separately) is applied during training only.
- **Optimization:** AdamW, linear warmup then cosine decay, grad-norm clip 1.0, optional
  bf16 autocast (`--amp`).

## Main arguments

| Argument | Default | Meaning |
|---|---|---|
| `--demo_datasets` | `./datasets/square_image_v15.hdf5` | Demonstration files (progress t/(T-1)) |
| `--random_datasets` | `./datasets/random/square_random.hdf5` | Random-action files (progress 0) |
| `--policy_datasets` | none | Policy rollouts from `collect_policy_data.py` (progress by outcome) |
| `--demo_frac` | 0.5 | Fraction of each batch drawn from demos |
| `--policy_frac` | 0.25 | Fraction drawn from policy rollouts (rest: random) |
| `--samples_per_epoch` | number of train frames | Frames drawn per epoch |
| `--val_frac` / `--split_seed` | 0.05 / 0 | Held-out episode fraction per source |
| `--epochs` / `--batch_size` / `--lr` | 30 / 64 / 3e-4 | |
| `--lambda_aux` / `--lambda_prog` | 0.5 / 1.0 | Loss weights |
| `--progress_loss` | `mse` | `bce` trains faster when predictions saturate |
| `--shift_pad` | 4 | Random-shift pad in pixels (0 disables) |
| `--amp` | off | bf16 autocast on CUDA |
| `--out` | `checkpoints/world_model_encoder/wm_encoder.pt` | Checkpoint path |
| `--wandb_mode` | `online` | `offline` or `disabled` to skip uploading |

## Outputs

- `--out`: the latest checkpoint. `<out>_best.pt` is the best by `val/loss`.
- Each checkpoint holds `encoder` (without DINOv2), `aux_head`, `progress_head`,
  `encoder_config`, `head_hidden`, `split` (sources and train/val episode names) and `epoch`.
- wandb project `world-model-encoder`: metrics, a code snapshot, and the checkpoints.

To load the encoder:

```python
ckpt = torch.load(path, weights_only=False)
enc = LatentEncoder(LatentEncoderConfig(**ckpt["encoder_config"]))
enc.load_state_dict(ckpt["encoder"], strict=False)  # DINOv2 weights come from torch.hub
```

## Metrics to watch

| Metric | Good run (Square, 30 epochs) | Meaning |
|---|---|---|
| `val_demo/progress_spearman` | ≈ 0.99 | g orders demo frames by progress |
| `val/progress_auroc` | ≈ 1.0 | g separates demo frames from random frames |
| `val_random/progress_pred_mean` | ≈ 0 | g stays low on random frames |
| `val_demo/nut_pos_rmse_m` | ≈ 3 mm | nut position recoverable from `z` |
| `val_demo/nut_quat_deg` | ≈ 3° | nut rotation recoverable from `z` |
| `val/nut_rel_pos_rmse_m` | ≈ 6 mm | nut position relative to the gripper |
| `val_policy/progress_auroc` | not measured yet | g separates frames of successful from failed rollouts (what ranking needs) |
| `val_policy/progress_pred_mean_success` / `_failure` | | mean g on successful vs failed rollout frames |

Metrics prefixed `val_demo/`, `val_random/` and `val_policy/` use only that source. `val/`
losses and aux errors use all held-out frames; `val/progress_*` uses demo and random frames
only, so that its AUROC stays a demo-vs-random number. `val_policy/progress_auroc` needs
held-out rollouts with both outcomes; with few policy episodes it may be missing. Two numbers look bad but are artifacts:

- `val/progress_spearman` is low (≈ 0.77) because all random frames share the label 0 and
  ties are ranked arbitrarily. Use `val_demo/progress_spearman`.
- `val_random/nut_pos_r2` can be negative because the nut barely moves in random episodes,
  so the variance R² divides by is tiny. Use `val_random/nut_pos_rmse_m`.

Gripper-pose metrics (`*eef*`) are near-perfect by construction because the gripper pose
is also in the proprio input.
