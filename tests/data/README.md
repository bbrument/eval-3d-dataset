# Test fixture

`lego/` is a reduced evaluation kit of the `lego` object of the [Martine dataset](https://doi.org/10.7910/DVN/2BEPEH),
used by [config/example.yaml](../../config/example.yaml) and `tests/test_example_fixture.py`:

- `gt/eval_cameras.sfm`: 12 of the 84 evaluation cameras, intrinsics rescaled to 1/8 resolution;
- `gt/masks_eval/`: the matching silhouettes, 1196×797;
- `gt/gt_pcd.npy`: the ground-truth point cloud resampled at 0.5 mm (the published one is at 0.05 mm);
- `pgsr_d4.ply`: a PGSR reconstruction from the d4 images, decimated to 40,000 faces.

Derived from the Martine dataset, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Cite the Martine paper (see the [README](../../README.md#citation)) when reusing them.
