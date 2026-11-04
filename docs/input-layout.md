# Input layout

The files the pipeline reads and writes, per object and method, and how it finds the GT mesh and the
cameras.

[← back to README](../README.md)

Every dataset uses the same on-disk shape. `{object}` is substituted from `dataset.objects`.

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
│   └── challenges/                          # optional (N,) bool masks, see challenges.md
│       ├── excluded.npy                     # True = never evaluated
│       └── <property>.npy                   # True = in the zone (e.g. lambertian.npy)
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

**Published Martine dataset.** The GT side is not rebuilt: `config/martine.yaml` reads the evaluation kit of
each object in place (read-only), and only `<method>/` folders are written under `eval_root`:

```
<dataset>/<object>/gt/
├── gt.ply                                   # GT mesh (mm, dataset frame)
├── gt_pcd.npy, gt_pcd_provenance.json       # the exact sampled GT cloud of the paper
├── attributes/, challenges/                 # as above, indexed on gt_pcd.npy
├── masks_eval/                              # full-resolution silhouettes used by cleanup
├── masks_watertight/                        # only for objects whose GT has holes
└── eval_cameras.sfm                         # the cameras of masks_eval/
```

Martine terms (resolution tiers `d1/d2/d4/d8`, MVS/MVPS captures) and the camera conventions are defined once, in
[martine-dataset](https://github.com/martine-benchmark/martine-dataset).

**GT mesh discovery.** Any `*.ply` in `Groundtruth/` is a candidate; names containing `clean`,
`pcd`, `point`, `sampled` or `downsample` are excluded as derived products, and the list is
sorted so the choice is deterministic. Name yours `gt_mesh.ply` and there is no ambiguity.

**Cameras.** Searched in order: `cameras.npz`, `sfm.json` (AliceVision SfMData),
`cameras.json`, `*.sfm`. All meshes must already share a frame — the pipeline does not
register.
