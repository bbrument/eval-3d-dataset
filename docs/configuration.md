# Configuration

[← back to README](../README.md)

Start from `config/martine.yaml` or `config/skoltech3d.yaml`. Both ship with placeholder
paths; edit the entries under `paths:` and nothing else is required.

For the published Martine dataset, `config/martine.yaml` also sets the optional read-only
inputs of the evaluation kit shipped under `<object>/gt/`: `paths.gt_artifacts_root` (sampled
GT `gt_pcd.npy`, `attributes/`, `challenges/`), `paths.eval_cameras` (`eval_cameras.sfm`),
`paths.eval_masks` (`masks_eval/`) and `paths.watertight_masks` (`masks_watertight/`; hole
culling runs only when this directory holds PNG masks). Left unset, the pipeline uses the
workspace `Groundtruth/` and the cameras/masks found under `paths.data_root`.

Every command takes `-c/--config`; it is required.

## The parameters that change your numbers

| key | unit | what it does |
|---|---|---|
| `evaluation.downsample_density` | mesh units | Resampling step for **both** clouds, so a denser mesh cannot buy a better score. Smaller = more points, slower, more memory. |
| `evaluation.sampling_seed` | integer | Seed for the GT down-sampling shuffle, so `gt_pcd.npy` is reproducible across runs. Default `42`. |
| `evaluation.max_dist` | mesh units | Distances above this are dropped from the Chamfer average, bounding gross outliers. **Its cost is reported as `coverage`** — if coverage is low, `max_dist` is doing too much work and the Chamfer is not meaningful. |
| `evaluation.fscore_thresholds` | mesh units | The reporting grid. `precision`/`recall`/`fscore` are index-aligned with `thresholds`. |
| `cleanup.dilation_radius` | **pixels** | Silhouette dilation before carving reconstructed vertices that project outside every mask. Larger = more forgiving at the border. `12` for both datasets here. |
| `cleanup.z_threshold` | mesh units | Drops reconstructed points below this height (support plane). `null` disables. |
| `dataset.num_views` | — | Camera views used for masking and visibility. |

## Masks

`Groundtruth/challenges/*.npy` are boolean arrays over the **GT point cloud**, `True` meaning
*drop this point*. They combine with logical OR, and the reconstruction side inherits the
decision through the nearest-neighbour index.

`excluded.npy` is applied automatically whenever the file exists. Whatever was applied is
recorded in `metrics["exclude_masks"]`, so a result file states what was done rather than what
was supposed to be done. Extra masks can be requested per run with `--extra-exclude <name>` on
`curves` and `recompute`.

## Data-to-GT alignment (optional)

Everything above assumes the reconstruction is already expressed in the ground-truth frame,
at the ground-truth scale. That holds for pipelines that reconstruct from the dataset's
calibrated cameras. It does **not** hold for feed-forward models such as VGGT, whose pointmaps
live in the first camera's frame at an arbitrary scale — evaluated as-is they score `NaN` or
near-zero coverage.

```yaml
alignment:
  enabled: true          # default false: nothing below runs, numbers are unchanged
  mode: sim3             # sim3 = rotation + translation + scale; se3 = no scale
  init: centroid         # centroid | cameras | null
  cameras_src: null      # init: cameras — camera file of the reconstruction ({object}/{method} ok)
  cameras_dst: null      # init: cameras — GT-frame cameras (default: paths.data_root cameras)
```

`eval-pipeline -c $CFG evaluate --align` / `--no-align` overrides `enabled` for one run.

**What it does.** Before any distance is computed, the cleaned mesh is registered onto
`gt_pcd.npy` with a robust point-to-point ICP (`src/core/alignment.py`): correspondences beyond
3× the median residual are rejected and the rest are Huber-weighted, so the estimate tolerates
a large fraction of spurious geometry. `init: centroid` matches centroids and RMS radii — enough
for moderate misorientation; for large rotations use `init: cameras`, which seeds the ICP with a
closed-form Umeyama fit on index-aligned camera centres (`.npy` of `(N,3)` centres or `(N,4,4)`
camera-to-world matrices, or any camera format the pipeline reads). The *aligned* mesh is then
sampled and measured exactly as before, so `downsample_density`, `max_dist` and the thresholds
keep their GT-unit meaning.

**What it writes.** `eval_results/alignment.json` holds the 4×4 transform applied (mesh → GT)
and the solver statistics — median residual before/after, inlier fraction, scale, iterations —
and `metrics.json` gains `alignment_residual_median`. Nothing is written when `enabled: false`.

**How to read the numbers afterwards.** Alignment is fitted *to the GT*, so it hands the
method the best rigid-or-similarity pose it can have. A Chamfer obtained this way measures
shape, not pose or scale, and is not comparable with un-aligned results — compare aligned cells
with aligned cells. `coverage` keeps its definition (share of points within `max_dist`) but
changes meaning slightly: after alignment a low `coverage` can no longer be blamed on a frame
mismatch, it reports genuinely missing or spurious geometry. Check `alignment_residual_median`
too — if it is not small against `max_dist`, the ICP did not find a good fit and the cell
should be treated as failed, whatever the Chamfer says.

## The units trap (sk3d)

`downsample_density`, `max_dist` and `fscore_thresholds` are in **the mesh's own units**. sk3d
meshes are in **metres**; the martine-style data is in **millimetres**. Left unconverted,
`downsample_density: 0.07` means a 70 mm step and `max_dist: 5.0` means five metres — no
filtering at all. **The metrics come out meaningless without failing.** Convert the meshes to
millimetres when building the eval tree and keep the thresholds in millimetres.
`dataset.gt_units` and `dataset.units_scale_to_mm` record the source units so the conversion is
auditable; the pipeline does not apply them itself.

sk3d also uses a looser `max_dist` (5.0 vs 2.0): its objects are much larger — the
`wooden_trex` GT diagonal is 434.9 mm against ~150 mm for DiLiGenT-MV — so a tighter bound
would clip genuine error.
