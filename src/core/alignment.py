"""Data-to-GT alignment: closed-form Umeyama and robust Sim(3)/SE(3) ICP.

The pipeline normally assumes every reconstruction already lives in the ground-truth
frame. Feed-forward models (VGGT-style pointmaps) do not: they are expressed in the
first camera's frame at an arbitrary scale. This module estimates the similarity
transform that brings such a reconstruction onto `gt_pcd` so the usual metrics can be
computed. It is opt-in — see `AlignmentConfig` and `pipeline.evaluate`.

Design notes
------------
- **Robust criterion.** Each ICP iteration keeps correspondences closer than
  `inlier_factor x median(distance)` (median-based rejection, immune to up to ~50% of
  gross outliers), then weights the survivors with a Huber kernel whose knee is that
  same median. Both are scale-free, so no absolute threshold in mesh units is needed
  before the scale is known.
- **Initialisation.** Either a caller-supplied 4x4 (typically Umeyama on camera
  centres), or centroid alignment plus RMS-radius scale matching. ICP is local; the
  centroid init handles moderate rotations but not, say, a 180-degree flip — for those,
  provide camera poses.
- **Bounds.** Iterations are capped (`max_iterations`) and stop early when the median
  residual changes by less than `tol` relatively. Inputs are subsampled to
  `max_src_points` / `max_dst_points` with a seeded RNG so the cost is bounded and
  the result reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

_MIN_POINTS = 3


def _as_points(pts: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}")
    if len(arr) < _MIN_POINTS:
        raise ValueError(f"{name} needs at least {_MIN_POINTS} points, got {len(arr)}")
    return arr


def _check_transform(T: np.ndarray, name: str = "init") -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got {T.shape}")
    return T


def _compose(scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Closed-form least-squares Sim(3) (or SE(3)) between matched point sets.

    Umeyama (1991), optionally weighted. Solves `dst ~= s R src + t` and returns the
    4x4 matrix `[[sR, t], [0, 1]]`. The reflection case (det < 0) is handled so the
    result is always a proper rotation.

    Args:
        src: Source points, shape (N, 3).
        dst: Target points, shape (N, 3), index-aligned with `src`.
        with_scale: Estimate a scale (Sim(3)) or fix it to 1 (SE(3)).
        weights: Optional non-negative per-correspondence weights, shape (N,).

    Returns:
        4x4 transform mapping `src` onto `dst`.
    """
    src = _as_points(src, "src")
    dst = _as_points(dst, "dst")
    if src.shape != dst.shape:
        raise ValueError(f"src and dst must have the same shape, got {src.shape} vs {dst.shape}")

    if weights is None:
        w = np.full(len(src), 1.0 / len(src))
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(src),):
            raise ValueError(f"weights must have shape ({len(src)},), got {w.shape}")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must not all be zero")
        w = w / total

    mu_src = w @ src
    mu_dst = w @ dst
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c * w[:, None]).T @ src_c  # (3, 3): sum_i w_i dst_i src_i^T
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = float(np.sum(w * np.sum(src_c**2, axis=1)))
        if var_src <= 0:
            raise ValueError("src points are all identical; scale is undefined")
        scale = float(np.trace(np.diag(D) @ S)) / var_src
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return _compose(scale, R, t)


def apply_transform(
    mesh_or_pts: trimesh.Trimesh | np.ndarray, T: np.ndarray
) -> trimesh.Trimesh | np.ndarray:
    """Apply a 4x4 transform to points (N, 3) or to a mesh. Never mutates its input.

    Args:
        mesh_or_pts: Either an (N, 3) array or a `trimesh.Trimesh`.
        T: 4x4 homogeneous transform.

    Returns:
        A new array or a new mesh (faces shared, vertices transformed).
    """
    T = _check_transform(T, "T")
    if isinstance(mesh_or_pts, trimesh.Trimesh):
        vertices = np.asarray(mesh_or_pts.vertices, dtype=np.float64)
        new_vertices = vertices @ T[:3, :3].T + T[:3, 3]
        return trimesh.Trimesh(vertices=new_vertices, faces=mesh_or_pts.faces.copy(), process=False)

    pts = np.asarray(mesh_or_pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    return pts @ T[:3, :3].T + T[:3, 3]


def _subsample(pts: np.ndarray, max_points: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_points is None or len(pts) <= max_points:
        return pts
    idx = rng.choice(len(pts), size=max_points, replace=False)
    return pts[idx]


def _centroid_init(src: np.ndarray, dst: np.ndarray, scale: bool) -> np.ndarray:
    """Centroid translation plus RMS-radius scale matching."""
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    if scale:
        r_src = np.sqrt(np.mean(np.sum((src - mu_src) ** 2, axis=1)))
        r_dst = np.sqrt(np.mean(np.sum((dst - mu_dst) ** 2, axis=1)))
        s = float(r_dst / r_src) if r_src > 0 else 1.0
    else:
        s = 1.0
    return _compose(s, np.eye(3), mu_dst - s * mu_src)


def _robust_weights(dist: np.ndarray, inlier_factor: float) -> tuple[np.ndarray, float]:
    """Median-based rejection followed by Huber weighting. Returns (weights, knee)."""
    knee = float(np.median(dist))
    if knee <= 0:
        # Degenerate (already exact): keep every point uniformly.
        return np.ones_like(dist), 0.0
    inliers = dist < inlier_factor * knee
    weights = np.where(dist <= knee, 1.0, knee / np.maximum(dist, 1e-300))
    weights = np.where(inliers, weights, 0.0)
    return weights, knee


def align_sim3(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    init: np.ndarray | None = None,
    scale: bool = True,
    robust: bool = True,
    max_iterations: int = 50,
    tol: float = 1e-6,
    inlier_factor: float = 3.0,
    max_src_points: int | None = 20_000,
    max_dst_points: int | None = 200_000,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Robust point-to-point ICP estimating a Sim(3) (or SE(3)) from `src` to `dst`.

    Args:
        src_pts: Source cloud (the reconstruction), shape (N, 3).
        dst_pts: Target cloud (the ground truth), shape (M, 3).
        init: Optional 4x4 initial transform (e.g. Umeyama on camera centres). Defaults
            to centroid alignment with RMS-radius scale matching.
        scale: Estimate scale (Sim(3)) or fix it to 1 (SE(3)).
        robust: Median-based rejection + Huber weights per iteration. When False every
            correspondence gets unit weight — plain ICP, sensitive to outliers.
        max_iterations: Hard cap on ICP iterations.
        tol: Stop when the median residual changes by less than this relative amount.
        inlier_factor: Correspondences farther than `inlier_factor x median` are rejected.
        max_src_points / max_dst_points: Subsample bounds (None = no bound).
        seed: RNG seed for the subsampling.

    Returns:
        (T, stats) with `T` the 4x4 transform mapping `src_pts` onto `dst_pts` and
        `stats` a JSON-serialisable dict: residual_median_before / _after, residual_rmse_after,
        inlier_fraction, scale, n_iterations, max_iterations, converged, n_src, n_dst.
    """
    src_full = _as_points(src_pts, "src_pts")
    dst_full = _as_points(dst_pts, "dst_pts")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    rng = np.random.default_rng(seed)
    src = _subsample(src_full, max_src_points, rng)
    dst = _subsample(dst_full, max_dst_points, rng)

    T = _check_transform(init) if init is not None else _centroid_init(src, dst, scale)
    tree = cKDTree(dst)

    def residuals(T_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        moved = apply_transform(src, T_cur)
        dist, idx = tree.query(moved, k=1, workers=-1)
        return dist, idx

    dist, idx = residuals(T)
    median_before = float(np.median(dist))
    prev_median = median_before
    # Absolute floor: a median residual below this is numerically zero for the cloud.
    abs_tol = tol * float(np.linalg.norm(dst.max(axis=0) - dst.min(axis=0)))
    n_iter = 0
    converged = False
    weights = np.ones(len(src))

    for n_iter in range(1, max_iterations + 1):
        if robust:
            weights, _ = _robust_weights(dist, inlier_factor)
        if np.count_nonzero(weights) < _MIN_POINTS:
            break

        T = umeyama(src, dst[idx], with_scale=scale, weights=weights)
        dist, idx = residuals(T)
        median = float(np.median(dist))

        if median <= abs_tol or abs(prev_median - median) <= tol * prev_median:
            converged = True
            break
        prev_median = median

    if robust:
        weights, _ = _robust_weights(dist, inlier_factor)
    inlier_mask = weights > 0
    s_est = float(np.cbrt(np.linalg.det(T[:3, :3])))

    stats = {
        "residual_median_before": median_before,
        "residual_median_after": float(np.median(dist)),
        "residual_rmse_after": float(np.sqrt(np.mean(dist[inlier_mask] ** 2)))
        if inlier_mask.any() else float("nan"),
        "inlier_fraction": float(inlier_mask.mean()),
        "scale": s_est,
        "n_iterations": int(n_iter),
        "max_iterations": int(max_iterations),
        "converged": bool(converged),
        "n_src": int(len(src)),
        "n_dst": int(len(dst)),
    }
    return T, stats


def load_camera_centers(path: str | Path) -> np.ndarray:
    """Load camera centres (N, 3) from a camera file for the `init: cameras` mode.

    Accepted formats:
        - `.npy`: either (N, 3) centres or (N, 4, 4) camera-to-world matrices.
        - `.npz` / `.json` / `.sfm`: the pipeline's own camera formats (see `core.camera`).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Camera file not found: {path}")

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr.astype(np.float64)
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr[:, :3, 3].astype(np.float64)
        raise ValueError(f"{path}: expected (N, 3) centres or (N, 4, 4) c2w matrices, got {arr.shape}")

    from .camera import get_camera_centers, load_cameras_auto

    cameras = load_cameras_auto(path)
    return get_camera_centers(cameras["Rt"]).astype(np.float64)


def camera_init(src_path: str | Path, dst_path: str | Path, with_scale: bool = True) -> np.ndarray:
    """Umeyama on index-aligned camera centres — the recommended ICP initialisation."""
    src = load_camera_centers(src_path)
    dst = load_camera_centers(dst_path)
    if src.shape != dst.shape:
        raise ValueError(
            f"Camera sets differ: {len(src)} centres in {src_path} vs {len(dst)} in {dst_path}. "
            "init: cameras needs the same views in the same order on both sides."
        )
    return umeyama(src, dst, with_scale=with_scale)
