"""Precision / recall / F-score pooled over objects, from the distances saved by `evaluate`.

Pooling concatenates the point-level distances of every object before counting, so an object weighs by its
number of points, as if all objects were one mesh. A failed object (no reconstruction) brings no data point and its
GT points stay in the pool, unreached: it lowers recall and F, not precision. Counts are exact and the ratios are float64 (core.metrics.compute_fscore_curve is float32 and
per object). A point counts when its distance is strictly below the threshold, as in core.metrics.
"""

from typing import Optional

import numpy as np

from ..config import Config
from ..core.masking import apply_exclude, load_gt_exclude


def prf_curves(data2gt: np.ndarray, gt2data: np.ndarray, thresholds: np.ndarray):
    """(precision, recall, fscore) at each threshold, as fractions."""
    thresholds = np.asarray(thresholds, dtype=np.float64)
    precision = np.searchsorted(np.sort(data2gt), thresholds, side="left") / len(data2gt)
    recall = np.searchsorted(np.sort(gt2data), thresholds, side="left") / len(gt2data)
    denom = precision + recall
    with np.errstate(invalid="ignore", divide="ignore"):
        fscore = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    return precision, recall, fscore


def load_distances(config: Config, obj: str, cell_dir: str, select: Optional[np.ndarray] = None):
    """(data2gt, gt2data) of one cell, GT `excluded` points removed, optionally restricted to `select` GT points.

    A reconstructed point follows its nearest GT point (data2gt_idx), as in `evaluate`.
    """
    dist = config.get_eval_dir(obj, cell_dir) / "distances"
    data2gt = np.load(dist / "data2gt_dist.npy")
    gt2data = np.load(dist / "gt2data_dist.npy")
    idx = np.load(dist / "data2gt_idx.npy")
    exclude, _ = load_gt_exclude(config.get_gt_artifacts_dir(obj), len(gt2data), verbose=False)
    if select is not None:
        if len(select) != len(gt2data):
            raise ValueError(f"{obj}/{cell_dir}: selection mask has length {len(select)} but the GT distances have "
                             f"{len(gt2data)} points. Stale mask or stale GT — refusing to guess.")
        drop = ~np.asarray(select, bool)
        exclude = drop if exclude is None else (exclude | drop)
    return apply_exclude(data2gt, gt2data, idx, exclude)


def missed_gt(config: Config, obj: str, select: Optional[np.ndarray] = None):
    """(data2gt, gt2data) of a failed object: no data point, every kept GT point at an infinite distance.

    Same GT points as `load_distances` (`excluded` removed, optionally restricted to `select`).
    """
    gt_dir = config.get_gt_artifacts_dir(obj)
    n_gt = len(np.load(gt_dir / "gt_pcd.npy", mmap_mode="r"))
    exclude, _ = load_gt_exclude(gt_dir, n_gt, verbose=False)
    keep = np.ones(n_gt, bool) if exclude is None else ~exclude
    if select is not None:
        if len(select) != n_gt:
            raise ValueError(f"{obj}: selection mask has length {len(select)} but the GT has {n_gt} points. "
                             "Stale mask or stale GT — refusing to guess.")
        keep &= np.asarray(select, bool)
    return np.empty(0), np.full(int(keep.sum()), np.inf)


def pooled_fscore(pairs: list, threshold: float) -> Optional[float]:
    """F-score in % of the concatenated (data2gt, gt2data) pairs; None when there is no GT point at all."""
    gt2data = [g for _, g in pairs if len(g)]
    if not gt2data:
        return None
    data2gt = [d for d, _ in pairs if len(d)]
    recall = np.mean(np.concatenate(gt2data) < threshold)
    precision = np.mean(np.concatenate(data2gt) < threshold) if data2gt else 0.0
    denom = precision + recall
    if denom == 0:
        return 0.0
    return float(2.0 * precision * recall / denom * 100.0)
