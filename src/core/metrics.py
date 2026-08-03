"""Distance and F-score metrics computation."""

import numpy as np
import sklearn.neighbors as skln


def compute_distances(pcd_a: np.ndarray, pcd_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute nearest neighbor distances from pcd_a to pcd_b.

    Args:
        pcd_a: Source point cloud, shape (N, 3).
        pcd_b: Target point cloud, shape (M, 3).

    Returns:
        Tuple of:
            - distances: Distance from each point in pcd_a to nearest in pcd_b, shape (N,)
            - indices: Index of nearest point in pcd_b for each point in pcd_a, shape (N,)
    """
    nn_engine = skln.NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    nn_engine.fit(pcd_b)

    distances, indices = nn_engine.kneighbors(pcd_a, n_neighbors=1, return_distance=True)

    return distances[:, 0].astype(np.float32), indices[:, 0].astype(np.int64)


def compute_chamfer(dist_a2b: np.ndarray, dist_b2a: np.ndarray, max_dist: float | None = None) -> dict:
    """Compute Chamfer distance.

    Args:
        dist_a2b: Distances from A to B, shape (N,).
        dist_b2a: Distances from B to A, shape (M,).
        max_dist: Optional max distance for outlier filtering.

    Returns:
        Dictionary with:
            - chamfer_a2b: Mean distance A to B (NaN if no point survives max_dist)
            - chamfer_b2a: Mean distance B to A (NaN if no point survives max_dist)
            - chamfer: Average of both directions (NaN propagates)
            - n_filtered_a2b: Number of filtered-out points (if max_dist set)
            - n_filtered_b2a: Number of filtered-out points (if max_dist set)
            - n_kept_a2b: Number of points that actually contributed to chamfer_a2b
            - n_kept_b2a: Number of points that actually contributed to chamfer_b2a
            - coverage_a2b: Fraction of A points within max_dist (contributing fraction)
            - coverage_b2a: Fraction of B points within max_dist
            - coverage: min(coverage_a2b, coverage_b2a) — the conservative summary

    Note on the empty-set case: returning 0.0 (the historical behaviour) awarded a
    *perfect* score to a reconstruction so misaligned that not a single point fell
    within max_dist. NaN is returned instead so the failure propagates into any mean
    and cannot be silently ranked as best. `coverage` exposes the *continuous* form of
    the same bias: a chamfer computed on 1% of the points is an artefact, not a score.
    """
    n_a_total = len(dist_a2b)
    n_b_total = len(dist_b2a)

    if max_dist is not None:
        dist_a2b_filt = dist_a2b[dist_a2b < max_dist]
        dist_b2a_filt = dist_b2a[dist_b2a < max_dist]
    else:
        dist_a2b_filt = dist_a2b
        dist_b2a_filt = dist_b2a

    n_kept_a2b = len(dist_a2b_filt)
    n_kept_b2a = len(dist_b2a_filt)
    n_filtered_a2b = n_a_total - n_kept_a2b
    n_filtered_b2a = n_b_total - n_kept_b2a

    chamfer_a2b = float(np.mean(dist_a2b_filt)) if n_kept_a2b > 0 else float("nan")
    chamfer_b2a = float(np.mean(dist_b2a_filt)) if n_kept_b2a > 0 else float("nan")
    chamfer = (chamfer_a2b + chamfer_b2a) / 2

    coverage_a2b = n_kept_a2b / n_a_total if n_a_total > 0 else float("nan")
    coverage_b2a = n_kept_b2a / n_b_total if n_b_total > 0 else float("nan")
    if n_a_total > 0 and n_b_total > 0:
        coverage = min(coverage_a2b, coverage_b2a)
    else:
        coverage = float("nan")

    return {
        "chamfer_a2b": chamfer_a2b,
        "chamfer_b2a": chamfer_b2a,
        "chamfer": chamfer,
        "n_filtered_a2b": n_filtered_a2b,
        "n_filtered_b2a": n_filtered_b2a,
        "n_kept_a2b": n_kept_a2b,
        "n_kept_b2a": n_kept_b2a,
        "coverage_a2b": coverage_a2b,
        "coverage_b2a": coverage_b2a,
        "coverage": coverage,
    }


def compute_fscore_curve(
    dist_a2b: np.ndarray,
    dist_b2a: np.ndarray,
    thresholds: list[float] | np.ndarray,
    max_dist: float | None = None,
) -> dict:
    """Compute precision, recall, and F-score at multiple thresholds.

    - Precision: proportion of reconstructed points within threshold of GT
    - Recall: proportion of GT points within threshold of reconstruction
    - F-score: harmonic mean of precision and recall

    Args:
        dist_a2b: Distances from reconstruction (A) to GT (B), shape (N,).
        dist_b2a: Distances from GT (B) to reconstruction (A), shape (M,).
        thresholds: List of distance thresholds.
        max_dist: Optional max distance for outlier filtering.

    Returns:
        Dictionary with:
            - thresholds: Array of thresholds
            - precision: Precision at each threshold
            - recall: Recall at each threshold
            - fscore: F-score at each threshold
    """
    thresholds = np.asarray(thresholds, dtype=np.float32)

    if max_dist is not None:
        dist_a2b = dist_a2b[dist_a2b < max_dist]
        dist_b2a = dist_b2a[dist_b2a < max_dist]

    n_a = len(dist_a2b)
    n_b = len(dist_b2a)

    precision = np.zeros(len(thresholds), dtype=np.float32)
    recall = np.zeros(len(thresholds), dtype=np.float32)
    fscore = np.zeros(len(thresholds), dtype=np.float32)

    # An empty filtered set means "not one point survived max_dist", which is a failure,
    # not a score of 0 on a well-defined denominator. NaN keeps it out of any mean and
    # out of any "best value" ranking. Note this is distinct from a legitimate 0.0,
    # which happens when points exist but none fall under threshold t.
    if n_a == 0 or n_b == 0:
        precision[:] = np.nan
        recall[:] = np.nan
        fscore[:] = np.nan
        return {
            "thresholds": thresholds,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }

    for i, t in enumerate(thresholds):
        prec = np.sum(dist_a2b < t) / n_a
        rec = np.sum(dist_b2a < t) / n_b

        if prec + rec > 0:
            f = 2 * prec * rec / (prec + rec)
        else:
            f = 0.0

        precision[i] = prec
        recall[i] = rec
        fscore[i] = f

    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def compute_metrics(
    dist_data2gt: np.ndarray,
    dist_gt2data: np.ndarray,
    thresholds: list[float] | np.ndarray,
    max_dist: float | None = None,
) -> dict:
    """Compute all metrics: Chamfer distance and F-score curve.

    Args:
        dist_data2gt: Distances from data to GT, shape (N,).
        dist_gt2data: Distances from GT to data, shape (M,).
        thresholds: F-score thresholds.
        max_dist: Max distance for outlier filtering.

    Returns:
        Dictionary with all metrics.
    """
    chamfer = compute_chamfer(dist_data2gt, dist_gt2data, max_dist)
    curves = compute_fscore_curve(dist_data2gt, dist_gt2data, thresholds, max_dist)

    return {
        **chamfer,
        **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in curves.items()},
    }
