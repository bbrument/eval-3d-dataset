# Configuration

[← back to README](../README.md)

Start from `config/example_martine.yaml` or `config/example_sk3d.yaml`. Both ship with relative
paths (resolved from the working directory); edit the two entries under `paths:` and nothing
else is required.

Every command takes `-c/--config`; it is required.

## The parameters that change your numbers

| key | unit | what it does |
|---|---|---|
| `evaluation.downsample_density` | mesh units | Resampling step for **both** clouds, so a denser mesh cannot buy a better score. Smaller = more points, slower, more memory. |
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
