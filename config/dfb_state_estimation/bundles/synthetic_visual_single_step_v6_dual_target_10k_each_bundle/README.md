# Synthetic Visual + Single-Step Bundle

This bundle definition describes the local upload set used to train the full shared-head single-step visual module.

Contents:

- `runs/dfb_state_estimation/synthetic_visual_dataset_fix_v6_20k_target_identity/target_fighter2/clean`
- `runs/dfb_state_estimation/synthetic_visual_dataset_fix_v6_20k_target_identity/target_fighter1/clean`
- `runs/dfb_state_estimation/synthetic_single_step_v6_target_fighter2_full10k`
- `runs/dfb_state_estimation/synthetic_single_step_v6_target_fighter1_full10k`

Upload policy:

- keep segmentation clean roots and single-step roots under the same repo-relative paths
- exclude audit-only artifacts from clean roots:
  - top-level `rgb/`
  - top-level `metadata/`
  - top-level `seg_color/`
  - per-sample `front_segmentation_color.ppm`
  - per-sample `rear_segmentation_color.ppm`
- keep full single-step labels and voting bundles intact

Rationale:

- `SyntheticSegmentationDataset` and `SyntheticSingleStepDataset` read only sample-local raw RGB, raw PGM segmentation, metadata, single-step labels, and voting bundles
- the excluded files are redundant for training and only inflate transfer size
