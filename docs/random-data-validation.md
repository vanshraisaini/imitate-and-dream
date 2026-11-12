# Random-action collector validation

The following checks were performed on Linux with Python 3.12.13, robosuite 1.5.2,
MuJoCo 3.3.1, PyTorch 2.9.1+cu128, and an RTX 5070 Ti Laptop GPU.

## Automated and simulator checks

- `python -m pytest -q`: **18 tests passed**. Coverage includes mutable observation
  buffers, sampled-action preservation, bounds, episode boundaries, collection
  limits, invalid observations/configuration, overwrite refusal, and partial files.
- Square reference-metadata smoke run: 2 episodes, 32 transitions.
- Lift JSON-config smoke run: 2 episodes, 16 transitions.
- Repeated Square smoke run with seed 42: all 30 saved HDF5 dataset arrays were
  exactly identical on this software/hardware stack.
- Both tasks passed camera shape/dtype, finite proprioception, action-bound and
  transition-continuity checks and loaded through the existing imitation loader.
- Collector help, Python compilation, `git diff --check`, and `pip check` passed.

## Initial training-data batch

```bash
python -m scripts.datasets.collect_random_data \
  --reference-dataset datasets/square_image_v15.hdf5 \
  --output datasets/random/square_random_10x400_seed42.hdf5 \
  --episodes 10 --steps-per-episode 400 --seed 42
```

This local batch contains 10 episodes and 4,000 transitions in a 65,819,726-byte
HDF5 file. Its SHA-256 is
`874c9082ab9765de8f7603749fbc743de85d7dc27fce2087784a1385066bc1ee`.
HDF5 bytes may differ on regeneration (including timestamp metadata); the hash
identifies the recorded local artifact, not a cross-run reproducibility promise.
Dataset binaries are not included in the repository.

Every episode passed image shape/dtype, required field, finite-value, action-bound,
count and transition-continuity checks. Saved actions exactly match the seed-42
uniform sampler. Robot, controller, control frequency and physics settings match
its reference. Each episode ends at the collection step limit; the source
configuration has `ignore_done=true`. No `.part` file remained. The existing
loader correctly padded action chunks at all ten episode boundaries. A two-sample
CUDA batch passed finite policy loss, backward, and prediction checks (2 x 16 x 7).
No optimizer step or model-training run was performed.

## Coverage measurements

End-effector positions were discretized into 2 cm world-coordinate cells using
`floor(position / 0.02)`. Counts compare 4,010 random states (including final next
observations) with 30,154 demonstration observations; sample counts are unequal.

| Measure | Result |
| --- | --- |
| Action histogram occupancy | All 20 equal-width bins occupied in each of 7 dimensions |
| Random occupied position cells | 1,465 |
| Demonstration occupied position cells | 1,902 |
| Shared position cells | 133 |
| Random cells absent from demonstrations | 1,332 |
| Demonstration cells visited by random | 6.99% |
| Random states outside demonstration axis-aligned position bounds | 74.41% |
| Maximum displacement from reset per episode | 0.183–0.524 m |
| Sum of recorded rewards | 0 in each episode |

The batch adds off-demonstration end-effector positions and has limited overlap
with the demonstrated task region. These are observed position/action statistics,
not object/contact coverage or evidence of improved world-model quality. No
independent success labels were collected. The batch complements demonstrations
and must not replace expert imitation targets. Dataset-size adequacy needs
subsequent world-model validation. The current imitation loader ignores
`next_obs`; a world-model loader is follow-up work.

Detailed local evidence is retained under `artifacts/setup/` and
`artifacts/issue-1/`, including collection commands/logs, source and output hashes,
installed packages, executable verification/analysis scripts, JSON reports and
visually inspected camera/coverage figures. These machine-specific records and
all generated datasets remain local and are excluded from the PR.
