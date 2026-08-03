"""Recompute metrics from saved distances without re-running evaluation.

The expensive half of `evaluate` (mesh upsampling + two KD-tree queries over ~10-30M
points) is already cached in `distances/*.npy`, at **full length, before any mask**.
Recomputing metrics is therefore pure numpy — seconds instead of ~20 minutes — and is
bit-for-bit equal to a full re-evaluation *provided the same exclusion mask is applied*.

That proviso used to be violated: this module ignored `challenges/excluded.npy` while
`evaluate.py` applied it, so recomputed metrics silently disagreed with evaluated ones
for the 21 of 27 objects that carry a mask. Fixed by routing both through
`core.masking`.
"""

import json
from pathlib import Path

import numpy as np

from ..config import Config
from ..core.metrics import compute_metrics, compute_fscore_curve
from ..core.masking import load_gt_exclude, apply_exclude


def recompute_metrics(
    config: Config,
    object_name: str,
    method_name: str,
    extra_exclude: list[str] | None = None,
    strict: bool = True,
) -> dict | None:
    """Recompute metrics.json and curves from saved distance arrays.

    Uses config.evaluation.max_dist and config.evaluation.fscore_thresholds.

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        extra_exclude: Additional GT exclusion masks (see core.masking).
        strict: Refuse to recompute when the cached arrays no longer match the current
            GT point count (stale cache). Such a cell needs a full `evaluate`, not a
            recompute, because its distances were measured against a GT that changed.

    Returns:
        Dict with metrics, or None if distances don't exist or the cache is stale.
    """
    eval_dir = config.get_eval_dir(object_name, method_name)
    gt_dir = config.get_gt_dir(object_name)
    distances_dir = eval_dir / "distances"

    d2g_path = distances_dir / "data2gt_dist.npy"
    g2d_path = distances_dir / "gt2data_dist.npy"
    idx_path = distances_dir / "data2gt_idx.npy"

    if not d2g_path.exists() or not g2d_path.exists():
        return None

    print(f"  Recomputing: {object_name}/{method_name}")

    dist_data2gt = np.load(d2g_path)
    dist_gt2data = np.load(g2d_path)

    # Cross-check against the *current* GT: the saved gt2data array has exactly one
    # entry per GT point, so a length mismatch means the GT changed since evaluation.
    # More reliable than mtimes on NFS.
    gt_pcd_path = gt_dir / "gt_pcd.npy"
    n_gt_current = None
    if gt_pcd_path.exists():
        n_gt_current = int(np.load(gt_pcd_path, mmap_mode="r").shape[0])
        if n_gt_current != len(dist_gt2data):
            print(
                f"    STALE CACHE: gt2data_dist has {len(dist_gt2data):,} entries but "
                f"current GT has {n_gt_current:,} points — needs full evaluate"
            )
            if strict:
                return None

    idx_data2gt = None
    if idx_path.exists():
        idx_data2gt = np.load(idx_path)

    exclude, applied_masks = load_gt_exclude(
        gt_dir, n_gt=len(dist_gt2data), extra_exclude=extra_exclude
    )
    if exclude is not None and idx_data2gt is None:
        print("    Cannot apply exclusion mask: data2gt_idx.npy missing — skipping cell")
        return None

    dist_data2gt, dist_gt2data = apply_exclude(
        dist_data2gt, dist_gt2data, idx_data2gt, exclude
    )

    metrics = compute_metrics(
        dist_data2gt,
        dist_gt2data,
        config.evaluation.fscore_thresholds,
        config.evaluation.max_dist,
    )

    gt_points_path = distances_dir / "gt_points.npy"
    data_points_path = distances_dir / "data_points.npy"
    if gt_points_path.exists():
        metrics["n_gt_points"] = int(np.load(gt_points_path, mmap_mode="r").shape[0])
    elif n_gt_current is not None:
        metrics["n_gt_points"] = n_gt_current
    if data_points_path.exists():
        metrics["n_data_points"] = int(np.load(data_points_path, mmap_mode="r").shape[0])
    metrics["seed"] = 42
    metrics["exclude_masks"] = [Path(p).name for p in applied_masks]
    metrics["max_dist"] = config.evaluation.max_dist

    # Preserve additive keys written by other stages, which are not recomputable
    # from distances and would otherwise be silently dropped.
    metrics_path = eval_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                previous = json.load(f)
            for key, value in previous.items():
                if key not in metrics:
                    metrics[key] = value
        except (json.JSONDecodeError, OSError) as e:
            print(f"    Could not merge previous metrics.json ({e}), writing fresh")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Curves must be rewritten too, or metrics.json ends up contradicting the arrays
    # next to it. But `curves` writes a *dense* sampling (100 thresholds) while the
    # metrics curve is the sparse report grid (5 thresholds): blindly writing the sparse
    # one here silently downgrades every densely-sampled cell. Preserve the existing
    # resolution instead — recompute the dense grid under the same mask.
    curves_dir = eval_dir / "curves"
    curves_dir.mkdir(exist_ok=True)
    thresholds_path = curves_dir / "thresholds.npy"

    curve_out = {
        "thresholds": np.array(metrics["thresholds"]),
        "precision": np.array(metrics["precision"]),
        "recall": np.array(metrics["recall"]),
        "fscore": np.array(metrics["fscore"]),
    }
    if thresholds_path.exists():
        existing_t = np.load(thresholds_path)
        if len(existing_t) > len(curve_out["thresholds"]):
            dense = compute_fscore_curve(
                dist_data2gt, dist_gt2data, existing_t, config.evaluation.max_dist
            )
            curve_out = {k: np.asarray(v) for k, v in dense.items()}
            print(f"    preserved dense curve resolution: {len(existing_t)} thresholds")

    for name, arr in curve_out.items():
        np.save(curves_dir / f"{name}.npy", arr)

    print(
        f"    chamfer={metrics['chamfer']:.4f} coverage={metrics['coverage']:.4f} "
        f"(max_dist={config.evaluation.max_dist})"
    )
    return metrics
