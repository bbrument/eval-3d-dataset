# Input layout

[← back to README](../README.md)

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
