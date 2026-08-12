# Changelog

## Unreleased

### Fixed
- **Reproducible GT point-cloud sampling.** `preprocess_gt` used to call
  `downsample_pcd` without a seed, so the shuffle inside it drew from
  `np.random.default_rng(None)` (OS entropy). Two runs on the same mesh produced
  `gt_pcd.npy` clouds that drifted by ~0.04%. The sampling seed is now configurable via
  the new `evaluation.sampling_seed` field (default `42`, matching the reconstruction-side
  seed already used in `pipeline/evaluate.py`) and is threaded into the GT downsampling.

  **Note for a future regeneration:** already-generated `gt_pcd.npy` files are *not*
  rewritten by this change. The first regeneration of any object under the default seed
  will therefore differ by ~0.04% from the cloud currently on disk (the old cloud was
  drawn under an unrecorded random seed). This is expected and harmless; the seed used is
  now recorded in `gt_pcd_provenance.json` (`sampling_seed`).
