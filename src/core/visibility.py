"""Visibility computation for point clouds using ray tracing."""

from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from tqdm import tqdm

from .camera import get_camera_centers, project_points


def load_mask(mask_path: str | Path) -> np.ndarray:
    """Load a binary mask image.

    Args:
        mask_path: Path to mask image (PNG).

    Returns:
        Binary mask array, shape (H, W).
    """
    img = Image.open(mask_path)
    mask = np.array(img)
    if mask.ndim == 3:
        mask = mask[..., 0]
    # Handle both uint8 (0/255) and boolean masks
    if mask.dtype == bool:
        return mask
    return mask > 127


def create_dilation_kernel(radius: int) -> np.ndarray:
    """Create a circular dilation kernel.

    Args:
        radius: Dilation radius in pixels.

    Returns:
        Binary kernel array.
    """
    diameter = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


# Default reference resolution for auto-dilation: Martine full resolution (9568 x 6376).
AUTO_DILATION_REF_PX = 48
AUTO_DILATION_REF_N_PIXELS = 9568 * 6376


def compute_auto_dilation(
    mask_n_pixels: int,
    ref_px: int = AUTO_DILATION_REF_PX,
    ref_n_pixels: int = AUTO_DILATION_REF_N_PIXELS,
) -> int:
    """Resolution-adaptive dilation radius for a mask.

    The dilation scales with the linear size of the mask (i.e. with the square root of
    its pixel count), so that a mask holds the same *physical* silhouette margin at any
    resolution. Calibrated by a reference point: ``ref_px`` pixels of dilation at a mask
    of ``ref_n_pixels`` total pixels (default 48 px at 9568x6376, Martine full res).

        dilation = round(ref_px * sqrt(mask_n_pixels / ref_n_pixels))

    With the default reference this gives 48 at full res, 24 at /2, 12 at /4, 6 at /8, and
    proportional values in between.

    Args:
        mask_n_pixels: Total number of pixels in the mask image (H * W).
        ref_px: Dilation radius (pixels) at the reference resolution.
        ref_n_pixels: Reference resolution as a total pixel count (H * W).

    Returns:
        Non-negative integer dilation radius (0 if inputs are non-positive).
    """
    if mask_n_pixels <= 0 or ref_n_pixels <= 0 or ref_px <= 0:
        return 0
    return int(round(ref_px * np.sqrt(mask_n_pixels / ref_n_pixels)))


def ray_visibility_check(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    camera_center: np.ndarray,
    ray_offset_factor: float = 0.01 / 100,
) -> np.ndarray:
    """Check visibility of points from a camera using ray tracing.

    Args:
        points: 3D points to check, shape (N, 3).
        mesh: Mesh for occlusion testing.
        camera_center: Camera center position, shape (3,).
        ray_offset_factor: Factor for ray origin offset to avoid self-intersection.

    Returns:
        Boolean array, True if point is visible (not occluded), shape (N,).
    """
    ray_directions = camera_center - points
    ray_norms = np.linalg.norm(ray_directions, axis=-1, keepdims=True)

    # Use per-point offset based on individual ray length (not global max)
    # This prevents numerical issues for points at varying distances
    ray_offsets = ray_offset_factor * ray_norms
    ray_directions_normalized = ray_directions / np.maximum(ray_norms, 1e-10)
    ray_origins = points + ray_offsets * ray_directions_normalized

    hit = mesh.ray.intersects_any(ray_origins, ray_directions_normalized)

    return ~hit


def check_points_in_bounds(
    points: np.ndarray,
    P: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Check if 3D points project inside image bounds.

    Args:
        points: 3D points, shape (N, 3).
        P: Projection matrix (4x4).
        image_shape: (height, width) of the image.

    Returns:
        Boolean array, True if point projects inside image bounds, shape (N,).
    """
    projected, depth_valid = project_points(points, P)

    h, w = image_shape
    x = projected[:, 0]
    y = projected[:, 1]

    in_bounds = depth_valid & (x >= 0) & (x < w) & (y >= 0) & (y < h) & ~np.isnan(x) & ~np.isnan(y)

    return in_bounds


def check_points_in_mask(
    points: np.ndarray,
    P: np.ndarray,
    mask: np.ndarray,
    dilation_radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Check if 3D points project inside a 2D mask.

    Args:
        points: 3D points, shape (N, 3).
        P: Projection matrix (4x4).
        mask: Binary mask, shape (H, W).
        dilation_radius: Optional mask dilation radius.

    Returns:
        Tuple of:
            - in_bounds: Boolean array, True if point is inside image bounds, shape (N,).
            - in_mask: Boolean array, True if point projects inside mask, shape (N,).
    """
    if dilation_radius is not None and dilation_radius > 0:
        kernel = create_dilation_kernel(dilation_radius)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    projected, depth_valid = project_points(points, P)

    h, w = mask.shape
    x = projected[:, 0]
    y = projected[:, 1]

    # Check bounds (also handles NaN from invalid depth)
    in_bounds = depth_valid & (x >= 0) & (x < w) & (y >= 0) & (y < h) & ~np.isnan(x) & ~np.isnan(y)

    in_mask = np.zeros(len(points), dtype=bool)
    valid_idx = np.where(in_bounds)[0]

    if len(valid_idx) > 0:
        xi = x[valid_idx].astype(int)
        yi = y[valid_idx].astype(int)
        in_mask[valid_idx] = mask[yi, xi]

    return in_bounds, in_mask


def compute_visibility_count(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    cameras: dict,
    masks_dir: str | Path,
    dilation_radius: int = 0,
    show_progress: bool = True,
    use_masks: bool = True,
) -> np.ndarray:
    """Compute per-point visibility count across all cameras.

    A point is visible from a camera if:
    1. It projects inside the (dilated) mask (if use_masks=True) or image bounds
    2. It is not occluded by the mesh

    Args:
        points: 3D points, shape (N, 3).
        mesh: Mesh for occlusion testing.
        cameras: Camera dict with 'P', 'Rt' arrays.
        masks_dir: Directory containing mask images (000.png, 001.png, ...).
        dilation_radius: Mask dilation radius in pixels.
        show_progress: Show progress bar.
        use_masks: Whether to use 2D masks for filtering.

    Returns:
        Visibility count per point, shape (N,), dtype int.
    """
    masks_dir = Path(masks_dir)
    n_views = len(cameras["P"])
    camera_centers = get_camera_centers(cameras["Rt"])

    visibility_count = np.zeros(len(points), dtype=np.int32)

    iterator = range(n_views)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing visibility")

    # Use mask_names from cameras dict if available (SfMData format)
    mask_names = cameras.get("mask_names")

    for view_idx in iterator:
        mask_path = None
        if mask_names is not None:
            mask_path = masks_dir / mask_names[view_idx]
        else:
            # Fallback: try sequential naming conventions
            for pattern in [f"{view_idx:03d}.png", f"V{view_idx:02d}.png"]:
                p = masks_dir / pattern
                if p.exists():
                    mask_path = p
                    break
        
        if not mask_path or not mask_path.exists():
            continue

        mask = load_mask(mask_path)
        P = cameras["P"][view_idx]
        camera_center = camera_centers[view_idx]

        if use_masks:
            _, in_view = check_points_in_mask(points, P, mask, dilation_radius)
        else:
            in_view = check_points_in_bounds(points, P, mask.shape)

        visible_from_cam = ray_visibility_check(points, mesh, camera_center)

        visible = in_view & visible_from_cam
        visibility_count += visible.astype(np.int32)

    return visibility_count


def estimate_normal_flip(
    oriented_dot_visible_front: int,
    oriented_dot_visible_total: int,
) -> bool:
    """Decide whether vertex normals must be flipped to point outward.

    Given, aggregated over all (vertex, view) pairs where the vertex is non-occluded
    and inside that view (the pairs the culling actually acts on), the number that are
    front-facing under the *un-flipped* normals, return True when fewer than half are
    front-facing -- i.e. the normal field points inward and must be negated so that
    ``n . (cam_center - p) > 0`` means "front-facing" as intended.

    A closed surface seen from a camera exposes (non-occluded) essentially only its
    front-facing vertices, so with correct outward normals this fraction is well above
    0.5; with inward normals it collapses below 0.5. The 0.5 boundary is therefore a
    wide, safe margin. Returns False when there is no visible evidence.
    """
    if oriented_dot_visible_total <= 0:
        return False
    return oriented_dot_visible_front < 0.5 * oriented_dot_visible_total


def filter_by_visibility(
    vertices: np.ndarray,
    mesh: trimesh.Trimesh,
    cameras: dict,
    masks_dir: str | Path,
    dilation_radius: int | str = 12,
    show_progress: bool = True,
    use_masks: bool = True,
    dilation_ref_px: int = AUTO_DILATION_REF_PX,
    dilation_ref_n_pixels: int = AUTO_DILATION_REF_N_PIXELS,
) -> np.ndarray:
    """Create a vertex mask keeping only vertices visible from at least one view.

    Uses strict mask filtering if use_masks=True: if a vertex projects inside a view's
    image bounds but outside the mask, it is removed.
    Otherwise (use_masks=False), it only checks for occlusion and camera bounds.

    Args:
        vertices: Mesh vertices, shape (N, 3).
        mesh: Mesh for occlusion testing.
        cameras: Camera dict.
        masks_dir: Masks directory.
        dilation_radius: Mask dilation radius in pixels (fixed int >= 0), or the string
            "auto" to derive a per-mask radius from its resolution via
            :func:`compute_auto_dilation` (dilation_ref_px / dilation_ref_n_pixels).
        show_progress: Show progress bar.
        use_masks: Whether to use 2D masks for filtering.
        dilation_ref_px: Auto-dilation reference radius (pixels at the reference res).
        dilation_ref_n_pixels: Auto-dilation reference resolution (total pixels).

    Returns:
        Boolean mask, True for visible vertices, shape (N,).
    """
    masks_dir = Path(masks_dir)
    n_views = len(cameras["P"])
    camera_centers = get_camera_centers(cameras["Rt"])
    n_vertices = len(vertices)

    # Dilation radius: fixed int, or per-mask "auto" scaling with each mask's resolution.
    auto_dilation = isinstance(dilation_radius, str) and dilation_radius == "auto"
    fixed_kernel = None
    if not auto_dilation:
        fixed_kernel = create_dilation_kernel(dilation_radius) if dilation_radius > 0 else None

    # Pre-load and dilate all masks
    masks = []
    image_shapes = []

    mask_names = cameras.get("mask_names")
    for view_idx in range(n_views):
        mask_path = None
        if mask_names is not None:
            mask_path = masks_dir / mask_names[view_idx]
        else:
            # Try multiple naming conventions
            for pattern in [f"{view_idx:03d}.png", f"V{view_idx:02d}.png"]:
                p = masks_dir / pattern
                if p.exists():
                    mask_path = p
                    break

        if mask_path and mask_path.exists():
            mask = load_mask(mask_path)
            image_shapes.append(mask.shape)
            if use_masks:
                if auto_dilation:
                    radius = compute_auto_dilation(mask.size, dilation_ref_px, dilation_ref_n_pixels)
                    kernel = create_dilation_kernel(radius) if radius > 0 else None
                    if show_progress and view_idx == 0:
                        print(f"    Auto-dilation: {radius}px for {mask.shape[1]}x{mask.shape[0]} masks "
                              f"(ref {dilation_ref_px}px @ {dilation_ref_n_pixels}px)")
                else:
                    kernel = fixed_kernel
                if kernel is not None:
                    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                masks.append(mask)
            else:
                masks.append(None)
        else:
            masks.append(None)
            image_shapes.append(None)

    # Pass 1: Find vertices inside at least one image's bounds
    if show_progress:
        print("  Pass 1: Finding vertices inside image bounds...")
    outside_all_images = np.ones(n_vertices, dtype=bool)

    for view_idx in range(n_views):
        if image_shapes[view_idx] is None:
            continue
        P = cameras["P"][view_idx]
        in_bounds = check_points_in_bounds(vertices, P, image_shapes[view_idx])
        outside_all_images[in_bounds] = False

    inside_any_image = ~outside_all_images
    if show_progress:
        print(f"    Vertices inside at least one image: {inside_any_image.sum()}/{n_vertices}")

    # Pass 2: Strict mask filtering
    # For each view, if vertex is inside bounds but outside mask, remove it
    if show_progress:
        print(f"  Pass 2: Mask filtering (use_masks={use_masks})...")
    keep_mask = inside_any_image.copy()

    if use_masks:
        iterator = range(n_views)
        if show_progress:
            iterator = tqdm(iterator, desc="  Filtering by masks")

        for view_idx in iterator:
            if masks[view_idx] is None:
                continue

            P = cameras["P"][view_idx]
            in_bounds, in_mask = check_points_in_mask(vertices, P, masks[view_idx], dilation_radius=0)

            # Keep if: (outside this view's bounds) OR (inside this view's mask)
            keep_this_view = ~in_bounds | in_mask
            keep_mask &= keep_this_view
    else:
        if show_progress:
            print("    Skipping mask filtering as use_masks=False")

    if show_progress:
        print(f"    Vertices after mask filtering: {keep_mask.sum()}/{n_vertices}")

    # Pass 3: Ray tracing for occlusion
    # Keep vertices visible (not occluded) from at least one view
    if show_progress:
        print("  Pass 3: Ray tracing for occlusion...")

    # Pre-compute which vertices are in mask for each view (needed for ray tracing)
    in_view_all_views = np.zeros((n_views, n_vertices), dtype=bool)
    for view_idx in range(n_views):
        if image_shapes[view_idx] is None:
            continue
        P = cameras["P"][view_idx]
        if use_masks and masks[view_idx] is not None:
            _, in_view = check_points_in_mask(vertices, P, masks[view_idx], dilation_radius=0)
        else:
            in_view = check_points_in_bounds(vertices, P, image_shapes[view_idx])
        in_view_all_views[view_idx] = in_view

    # Compute ray visibility for all views
    hits_all_views = np.zeros((n_views, n_vertices), dtype=bool)

    iterator = range(n_views)
    if show_progress:
        iterator = tqdm(iterator, desc="  Ray tracing")

    for view_idx in iterator:
        if image_shapes[view_idx] is None:
            continue
        camera_center = camera_centers[view_idx]
        visible_from_cam = ray_visibility_check(vertices, mesh, camera_center)
        hits_all_views[view_idx] = ~visible_from_cam

    # A vertex is visible if: not hit (not occluded) AND in mask (or bounds) for that view
    visible_per_view = ~hits_all_views & in_view_all_views
    visible_any = visible_per_view.any(axis=0)

    # Final mask: passed mask filtering AND visible from at least one view
    final_mask = keep_mask & visible_any

    if show_progress:
        print(f"    Final visible vertices: {final_mask.sum()}/{n_vertices}")

    return final_mask


def filter_by_issue_watertight(
    vertices: np.ndarray,
    mesh: trimesh.Trimesh,
    cameras: dict,
    masks_issue_watertight_dir: str | Path,
    visible_mask: np.ndarray | None = None,
    sample_count: int = 100,
    orientation_ratio_threshold: float = 0.7,
    show_progress: bool = True,
) -> np.ndarray:
    """Count, per vertex, how many watertight-issue masks flag it (Robin's culling).

    Ported from ``robin_voxel_vs_nn/src/core/visibility.py::filter_mesh_by_issue_watertight``.
    The watertight-issue masks (one per view, named ``000000.png``.. in glob order,
    matched to ``cameras["P"][i]``) mark the image region where the GT has a *hole*
    (wood: underside; sneakers: interior). A reconstruction vertex is flagged in a view
    when it (a) projects inside that view's issue mask, (b) is front-facing there
    (oriented normal . (cam_center - p) > 0), and (c) is non-occluded (ray visibility).
    Cleanup then drops any vertex flagged in >= 1 view: these are exactly the surfaces
    that fill in the GT's holes.

    Normal orientation: sample ``sample_count`` mesh points visible from camera 0 and
    measure the fraction whose normal faces the camera; the global flip decision reuses
    :func:`estimate_normal_flip` (fed with that front/total tally), so the outward
    convention matches the rest of the culling code.

    Args:
        vertices: Mesh vertices, shape (N, 3).
        mesh: Reconstruction mesh (for normals + occlusion).
        cameras: Camera dict with 'P', 'Rt'.
        masks_issue_watertight_dir: Directory of per-view issue masks (PNG, glob order = view order).
        visible_mask: Optional (N,) bool restricting work to already-visible vertices.
        sample_count: Points sampled from cam 0 to estimate normal orientation.
        orientation_ratio_threshold: (kept for signature parity / logging; the flip
            decision itself defers to estimate_normal_flip).
        show_progress: Print progress.

    Returns:
        issue_counts: (N,) int32, number of views flagging each vertex.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    n = len(vertices)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    if visible_mask is not None:
        visible_mask = np.asarray(visible_mask, dtype=bool)
        if visible_mask.shape[0] != n:
            raise ValueError(f"Expected visible_mask of length {n}, got {visible_mask.shape[0]}")
        active_vertices = visible_mask
    else:
        active_vertices = np.ones(n, dtype=bool)
    if not np.any(active_vertices):
        return np.zeros(n, dtype=np.int32)

    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if normals.shape[0] != n:
        if len(normals) == 0:
            return np.zeros(n, dtype=np.int32)
        raise ValueError(f"watertight_culling: expected {n} vertex normals, got {normals.shape[0]}")
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)

    n_views = len(cameras.get("P", []))
    if n_views == 0:
        return np.zeros(n, dtype=np.int32)
    camera_centers = get_camera_centers(cameras["Rt"])

    # --- Estimate normal orientation from points visible in camera 0, then reuse
    #     estimate_normal_flip for the global flip decision. ---
    oriented_normals = normals
    camera_center0 = camera_centers[0]
    sample_points = mesh.sample(max(sample_count * 5, sample_count))
    valid_samples = []
    for point in sample_points:
        if len(valid_samples) >= sample_count:
            break
        point = np.asarray(point, dtype=np.float32)
        if ray_visibility_check(point[None, :], mesh, camera_center0)[0]:
            valid_samples.append(point)

    if valid_samples:
        sp = np.asarray(valid_samples, dtype=np.float32)[:sample_count]
        sample_normals = normals[mesh.kdtree.query(sp)[1]]
        cam_dirs = camera_center0 - sp  # toward camera
        cam_dirs = cam_dirs / np.maximum(np.linalg.norm(cam_dirs, axis=1, keepdims=True), 1e-10)
        dots = np.einsum("ij,ij->i", sample_normals, cam_dirs)
        n_front = int(np.sum(dots > 0))
        n_total = int(len(dots))
        flip = estimate_normal_flip(n_front, n_total)
        if show_progress:
            frac = (n_front / n_total) if n_total else float("nan")
            print(f"  [watertight] front-facing fraction (cam0 samples)={frac:.3f} (flip={flip}, thr={orientation_ratio_threshold})")
        if flip:
            oriented_normals = -normals

    issue_mask_paths = sorted(Path(masks_issue_watertight_dir).glob("*.png"))
    if not issue_mask_paths:
        issue_mask_paths = sorted(Path(masks_issue_watertight_dir).glob("*.npy"))
    if not issue_mask_paths:
        print("  [watertight] no issue masks found; nothing culled")
        return np.zeros(n, dtype=np.int32)
    if show_progress:
        print(f"  [watertight] processing {len(issue_mask_paths)} issue masks over {n_views} views")

    counts = np.zeros(n, dtype=np.int32)
    iterator = enumerate(issue_mask_paths)
    if show_progress:
        iterator = tqdm(list(iterator), desc="  Watertight culling")
    for view_idx, mask_path in iterator:
        if view_idx >= n_views:
            break
        if not mask_path.exists():
            continue
        mask = load_mask(mask_path)
        P = cameras["P"][view_idx]
        _, in_mask = check_points_in_mask(vertices, P, mask, dilation_radius=0)

        camera_center = camera_centers[view_idx]
        cam_dirs = camera_center - vertices
        cam_dirs = cam_dirs / np.maximum(np.linalg.norm(cam_dirs, axis=1, keepdims=True), 1e-10)
        normal_dot = np.einsum("ij,ij->i", oriented_normals, cam_dirs)
        facing_camera = normal_dot > 0

        candidate_mask = active_vertices & in_mask & facing_camera
        candidate_indices = np.flatnonzero(candidate_mask)
        if len(candidate_indices) > 0:
            visible_from_cam = ray_visibility_check(vertices[candidate_indices], mesh, camera_center)
            counts[candidate_indices[visible_from_cam]] += 1

    if show_progress:
        print(f"  [watertight] finished: {int((counts > 0).sum())} vertices flagged, {int(counts.sum())} vertex-mask hits")
    return counts
