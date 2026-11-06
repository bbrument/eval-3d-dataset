# Configuration

The config keys that change your numbers, the evaluation masks, and the units trap. Every key and its
default is in `config/default.yaml`.

[← back to README](../README.md)

Start from your dataset's config in `config/` ([datasets.md](datasets.md)). Every bundled config ships with
placeholder paths; edit the entries under `paths:` and nothing else is required.

For the published Martine dataset, `config/martine.yaml` also sets the optional read-only
inputs of the evaluation kit shipped under `<object>/gt/`: `paths.gt_artifacts_root` (sampled
GT `gt_pcd.npy`, `attributes/`, `challenges/`), `paths.eval_cameras` (`eval_cameras.sfm`),
`paths.eval_masks` (`masks_eval/`) and `paths.watertight_masks` (`masks_watertight/`; hole
culling runs only when this directory holds PNG masks). Left unset, the pipeline uses the
workspace `Groundtruth/` and the cameras/masks found under `paths.data_root`. These templates start with
`{dataset_root}`, so `paths.dataset_root` (the download) and `paths.eval_root` (your workspace) are the two
keys to set. Relative paths are resolved from the directory you run `eval-pipeline` in.

Every command takes `-c/--config`; it is required.

## The parameters that change your numbers

| key | unit | what it does |
|---|---|---|
| `evaluation.downsample_density` | mesh units | Resampling step for **both** clouds, so a denser mesh cannot buy a better score. Smaller = more points, slower, more memory. |
| `evaluation.sampling_seed` | integer | Seed for the GT down-sampling shuffle, so `gt_pcd.npy` is reproducible across runs. Default `42`. |
| `evaluation.max_dist` | mesh units | Distances above this are dropped from the Chamfer average, bounding gross outliers. **Its cost is reported as `coverage`**: if coverage is low, `max_dist` is doing too much work and the Chamfer is not meaningful. |
| `evaluation.fscore_thresholds` | mesh units | The reporting grid. `precision`/`recall`/`fscore` are index-aligned with `thresholds`. |
| `evaluation.curvature_radius` | mesh units | Curvature of `attributes/curvature_values.npy` (Voronoi Covariance Measure): radius r of the connected ball, so the far side of a thin wall is not mixed in. Default `1.0` (`null` = `1.0`). Convex < 0. |
| `evaluation.curvature_tube` | mesh units | VCM tube thickness R. Default `0.5`. |
| `cleanup.dilation_radius` | **pixels** | Silhouette dilation before carving reconstructed vertices that project outside every mask. Larger = more forgiving at the border. Martine uses `auto` (48 px at full resolution, rescaled to each mask resolution); the other configs use `12`. |
| `cleanup.z_threshold` | mesh units | Drops reconstructed points below this height (support plane). `null` disables. |
| `dataset.num_views` | count | Camera views used for masking and visibility. |

## Masks

`challenges/*.npy` are boolean arrays over the **GT point cloud**. Two kinds share the folder:
`excluded.npy`, where `True` means *drop this point* (unreliable GT), and property masks such as
`lambertian.npy`, where `True` means *in this zone* (see [challenges.md](challenges.md)). Exclusion masks
combine with logical OR, and the reconstruction side inherits the decision through the nearest-neighbour
index.

`excluded.npy` is applied automatically whenever the file exists. Whatever was applied is
recorded in `metrics["exclude_masks"]`, so a result file states what was done rather than what
was supposed to be done. Extra masks can be requested per run with `--extra-exclude <name>` on
`curves` and `recompute`.

## The units trap (skoltech3d)

`downsample_density`, `max_dist` and `fscore_thresholds` are in **the mesh's own units**. skoltech3d
meshes are in **metres**; the martine-style data is in **millimetres**. Left unconverted,
`downsample_density: 0.07` means a 70 mm step and `max_dist: 5.0` means five metres, so no
filtering at all. **The metrics come out meaningless without failing.** Convert the meshes to
millimetres when building the eval tree and keep the thresholds in millimetres.
`dataset.gt_units` and `dataset.units_scale_to_mm` record the source units so the conversion is
auditable; the pipeline does not apply them itself.

skoltech3d also uses a looser `max_dist` (5.0 vs 2.0): its objects are much larger (the
`wooden_trex` GT diagonal is 434.9 mm against ~150 mm for DiLiGenT-MV), so a tighter bound
would clip genuine error.
