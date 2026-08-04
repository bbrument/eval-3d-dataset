# Reading `metrics.json`

[← back to README](../README.md)

Each object×method cell writes one `eval_results/metrics.json`.

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
it were perfect. See [CHANGELOG entry 1](../CHANGELOG.md) for why this matters.

Dense curves live in `eval_results/curves/` as four aligned arrays (`thresholds`, `precision`,
`recall`, `fscore`), independent of the coarse reporting grid.
