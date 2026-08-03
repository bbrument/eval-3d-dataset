# eval-3d-dataset

Evaluation pipeline for multi-view 3D surface reconstruction benchmarks.

It takes a ground-truth mesh and one reconstructed mesh per method, and produces Chamfer
distance, precision/recall/F-score at fixed thresholds, dense PR curves, a `coverage` figure
telling you how much of the cloud those numbers were actually computed on, and optional metric
visualisations.

The repository drives two datasets: a **millimetre-scale 84-view benchmark** (*martine* below)
and **Skoltech3D** (*sk3d*). Ready-to-edit configurations for both are in `config/`.

---

## ⚠️ Results from earlier versions are not comparable

If you evaluated with a copy of this pipeline from before **July 2026**, your numbers differ
from those produced today, and the difference is not a rounding matter.

**1. Chamfer distance returned `0.0` when no point survived `max_dist`.**
An empty filtered set means not one point of the reconstruction landed within `max_dist` of the
ground truth — a total failure. `0.0` is the *best possible* Chamfer, so those failures sorted
first and the zero propagated into every average taken over objects or methods. It now returns
`NaN`, which cannot be silently ranked as best. This is distinct from a legitimate `0.0`, which
is what you get when points exist but none fall under threshold *t*.

**2. `coverage` did not exist.** A Chamfer computed on 1% of the points is an artefact, not a
score, and the output gave you no way to see it. `coverage` is now reported next to every
Chamfer — see the field table below.

**3. Exclusion masks were applied inconsistently.** The masking logic lived inline in the
evaluation path only; the curve and recompute paths skipped it. Curves on disk could therefore
disagree with the `metrics.json` beside them for every object carrying a
`challenges/excluded.npy`. All three paths now share `src/core/masking.py`.

**4. Dense F-score curves were overwritten by the coarse reporting grid.** A curve computed on
100 thresholds could be rewritten on the 5 reporting thresholds, silently, so plots regenerated
afterwards were a coarse resampling of themselves.

Cross-version results should be regenerated, not reconciled.

---

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
pip install -e ".[visualization]"   # only for `visualize` (needs OSMesa)
```

Python ≥ 3.10. Dependencies: numpy, trimesh, scikit-learn, OpenCV, pydantic v2, click, PyYAML.
**No GPU is required** for the metrics.

The real constraint is memory: nearest-neighbour queries run over 10–30 M points and take
roughly 20 minutes per object×method cell.

Every command takes `-c/--config`; it is required.

```bash
eval-pipeline -c config/example_martine.yaml --help      # installed entry point
python -m src.cli -c config/example_martine.yaml --help  # equivalent, without installing
```

---

## Input layout

Both datasets use the same on-disk shape. `{object}` is substituted from `dataset.objects`.

```
<eval_root>/                                 # paths.eval_root, e.g. ./eval/{object}
├── Groundtruth/
│   ├── gt_mesh.ply                          # the GT mesh — you supply this
│   ├── gt_cleaned.ply                       # written by preprocess-gt --clean-gt
│   ├── gt_pcd.npy                           # written by preprocess-gt: (N,3) float
│   ├── gt_pcd_provenance.json               # source mesh, density, N, timestamp
│   ├── attributes/
│   │   ├── curvature_values.npy             # (N,) float32
│   │   └── visibility_count.npy             # (N,) int32 — cameras seeing each point
│   └── challenges/                          # optional (N,) bool masks, True = drop
│       └── excluded.npy
└── <method>/                                # one per entry in dataset.methods
    ├── results_raw/mesh.ply                 # your reconstruction — the only input you supply
    ├── results_cleaned/mesh.ply             # written by cleanup
    └── eval_results/                        # written by evaluate / curves
        ├── metrics.json
        ├── curves/{thresholds,precision,recall,fscore}.npy
        └── distances/{data2gt,gt2data}_{dist,idx}.npy, {data,gt}_points.npy

<data_root>/                                 # paths.data_root, e.g. ./data/{object}/images
├── masks/*.png                              # binary silhouettes, one per view
└── sfm.json                                 # cameras
```

Every `.npy` under `attributes/` and `challenges/` has exactly `N` entries and is indexed on
`gt_pcd.npy` — **not** on the mesh vertices. If you distribute these arrays alongside a GT
mesh, ship `gt_pcd.npy` with them or they index nothing.

**GT mesh discovery.** Any `*.ply` in `Groundtruth/` is a candidate; names containing `clean`,
`pcd`, `point`, `sampled` or `downsample` are excluded as derived products, and the list is
sorted so the choice is deterministic. Name yours `gt_mesh.ply` and there is no ambiguity.

**Cameras.** Searched in order: `cameras.npz`, `sfm.json` (AliceVision SfMData),
`cameras.json`, `*.sfm`. All meshes must already share a frame — the pipeline does not
register.

---

## Configuration

Start from `config/example_martine.yaml` or `config/example_sk3d.yaml`. Both ship with relative
paths (resolved from the working directory); edit the two entries under `paths:` and nothing
else is required.

### The parameters that change your numbers

| key | unit | what it does |
|---|---|---|
| `evaluation.downsample_density` | mesh units | Resampling step for **both** clouds, so a denser mesh cannot buy a better score. Smaller = more points, slower, more memory. |
| `evaluation.max_dist` | mesh units | Distances above this are dropped from the Chamfer average, bounding gross outliers. **Its cost is reported as `coverage`** — if coverage is low, `max_dist` is doing too much work and the Chamfer is not meaningful. |
| `evaluation.fscore_thresholds` | mesh units | The reporting grid. `precision`/`recall`/`fscore` are index-aligned with `thresholds`. |
| `cleanup.dilation_radius` | **pixels** | Silhouette dilation before carving reconstructed vertices that project outside every mask. Larger = more forgiving at the border. `12` for both datasets here. |
| `cleanup.z_threshold` | mesh units | Drops reconstructed points below this height (support plane). `null` disables. |
| `dataset.num_views` | — | Camera views used for masking and visibility. |

### Masks

`Groundtruth/challenges/*.npy` are boolean arrays over the **GT point cloud**, `True` meaning
*drop this point*. They combine with logical OR, and the reconstruction side inherits the
decision through the nearest-neighbour index.

`excluded.npy` is applied automatically whenever the file exists. Whatever was applied is
recorded in `metrics["exclude_masks"]`, so a result file states what was done rather than what
was supposed to be done. Extra masks can be requested per run with `--extra-exclude <name>` on
`curves` and `recompute`.

### The units trap (sk3d)

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

---

## End to end

```bash
CFG=config/example_martine.yaml     # or config/example_sk3d.yaml

# 1. Ground truth: clean the mesh, sample it, compute per-point attributes.
#    Writes Groundtruth/{gt_cleaned.ply,gt_pcd.npy,gt_pcd_provenance.json,attributes/}.
eval-pipeline -c $CFG preprocess-gt --clean-gt

# 2. Carve reconstructed vertices falling outside every silhouette.
#    Reads results_raw/mesh.ply, writes results_cleaned/mesh.ply.
eval-pipeline -c $CFG cleanup

# 3. Distances, Chamfer, coverage, F-score at the reporting thresholds.
#    Writes eval_results/metrics.json and eval_results/distances/.
eval-pipeline -c $CFG evaluate

# 4. Dense PR curves from the saved distances (no re-query).
#    Writes eval_results/curves/.
eval-pipeline -c $CFG curves -n 100

# Optional: rebuild metrics.json + curves from saved distances, e.g. after changing
# max_dist or adding a mask. Never re-runs the nearest-neighbour search.
eval-pipeline -c $CFG recompute
```

Scope any stage with `-o/--object` and `-m/--method`; re-run with `-f/--force`.

`preprocess-gt --clean-gt` removes ground-truth points that **no camera sees**, and resamples
from the cleaned surface. For a solid object this changes almost nothing; for an open,
foliage-like object it removes the entire unobservable interior. That is intended — scoring a
reconstruction on surfaces the rig never observed measures nothing — but it means GT point
counts are not comparable across the change. `gt_pcd_provenance.json` records which mesh the
cloud came from, the sampling density and the resulting count.

---

## Reading `metrics.json`

| field | meaning |
|---|---|
| `chamfer_a2b` | mean distance reconstruction → GT over points within `max_dist`. **`NaN` if none survived.** |
| `chamfer_b2a` | mean distance GT → reconstruction, same rule. |
| `chamfer` | mean of both directions; `NaN` propagates. |
| `coverage_a2b` | fraction of reconstruction points within `max_dist` — the share that contributed. |
| `coverage_b2a` | same, GT direction. |
| `coverage` | `min` of the two. **Read this before the Chamfer.** High accuracy at 20% coverage describes a fragment, not a reconstruction. |
| `n_kept_a2b` / `n_kept_b2a` | point counts behind each Chamfer. |
| `n_filtered_a2b` / `n_filtered_b2a` | points dropped by `max_dist`. |
| `thresholds` | reporting grid, echoing `evaluation.fscore_thresholds`. |
| `precision`, `recall`, `fscore` | arrays index-aligned with `thresholds`. All `NaN` if either cloud is empty. |
| `n_gt_points` / `n_data_points` | cloud sizes after resampling and masking. |
| `exclude_masks` | the mask files actually applied, by name. |
| `max_dist` | the value used, echoed so the file is self-describing. |
| `seed` | resampling seed (42). |

**`NaN` is a result, not a crash.** It means the cell produced nothing measurable at this
`max_dist`, and it is deliberately not `0.0` so it cannot be averaged into a ranking as though
it were perfect.

Dense curves live in `eval_results/curves/` as four aligned arrays (`thresholds`, `precision`,
`recall`, `fscore`), independent of the coarse reporting grid.

---

## Running on a cluster

Set `execution.mode: "slurm"` and fill `execution.slurm.account` / `partition`. Job templates in
`slurm/templates/` are fully parameterised — nothing site-specific is baked into `src/`.

`watch` polls for new meshes and submits jobs as they appear; `watch-status` reports on them.

> Helper scripts under `scripts/` were written for one specific machine and still contain
> absolute paths. They are convenience wrappers, not part of the pipeline — `src/` contains no
> absolute path.

---

## Other bundled configs

`config/` also carries working configurations for DiLiGenT-MV (`dlmv.yaml`), DTU (`dtu.yaml`),
LUCES-MV (`lucesmv.yaml`) and EvalMVX (`evalmvx.yaml`). These are live configurations rather
than templates and may contain machine-specific paths; use `example_martine.yaml` /
`example_sk3d.yaml` as your starting point.

---

## Tests

```bash
python -m pytest tests/ -q
```

The four `tests/core/test_rendering.py` cases need an OpenGL/OSMesa context and fail in a
headless environment. Everything else must pass.

---

## Scope

This pipeline evaluates **geometry**: Chamfer distance, precision/recall/F-score, coverage.

It does **not** evaluate surface orientation. The normal-map / Mean-Angular-Error path was
removed in August 2026: it had never produced a reported number in any results file, and it was
the only component requiring an unpublished internal library.

---

## Licence

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Baptiste Brument.
