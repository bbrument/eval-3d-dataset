"""Mesh cleanup via visibility-based filtering."""

from pathlib import Path

import numpy as np

from ..config import Config
from ..core.camera import load_cameras_auto
from ..core.mesh import load_mesh, save_mesh, filter_mesh_by_vertex_mask
from ..core.visibility import filter_by_visibility, filter_by_issue_watertight


def cleanup_mesh(
    config: Config,
    object_name: str,
    method_name: str,
    force: bool = False,
) -> None:
    """Clean up a reconstructed mesh by removing non-visible vertices.

    Algorithm:
    1. Load raw mesh from results_raw/mesh.ply
    2. For each camera view:
       - Project vertices to image
       - Check if inside (dilated) mask
       - Check visibility via ray tracing
    3. Keep vertices visible from at least one view
    4. Save cleaned mesh to results_cleaned/mesh.ply

    Args:
        config: Pipeline configuration.
        object_name: Object name.
        method_name: Method name.
        force: Overwrite existing output.
    """
    raw_mesh_path = config.get_raw_mesh_path(object_name, method_name)
    cleaned_mesh_path = config.get_cleaned_mesh_path(object_name, method_name)
    cameras_path = config.get_cameras_path(object_name, method_name)
    masks_dir = config.get_masks_dir(object_name, method_name)

    if not raw_mesh_path.exists():
        raise FileNotFoundError(f"Raw mesh not found: {raw_mesh_path}")

    if cleaned_mesh_path.exists() and not force:
        print(f"  Cleaned mesh exists, skipping: {cleaned_mesh_path}")
        return

    print(f"Cleaning mesh: {object_name}/{method_name}")

    mesh = load_mesh(raw_mesh_path)
    print(f"  Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print(f"  Warning: Skipping {object_name}/{method_name} - Empty mesh (0 vertices or 0 faces)")
        return

    if not cameras_path.exists():
        raise FileNotFoundError(f"Cameras not found: {cameras_path}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    # Apply z_threshold FIRST (before visibility) - matching reference implementation
    if config.cleanup.z_threshold is not None:
        z_mask = mesh.vertices[:, 2] >= config.cleanup.z_threshold
        n_before = len(mesh.vertices)
        mesh = filter_mesh_by_vertex_mask(mesh, z_mask)
        print(f"  After z-threshold ({config.cleanup.z_threshold}): {len(mesh.vertices)} vertices ({n_before - len(mesh.vertices)} removed)")

    cameras = load_cameras_auto(cameras_path)
    print(f"  Loaded {len(cameras['P'])} camera views")

    print(f"  Filtering by visibility (dilation={config.cleanup.dilation_radius}, use_masks={config.cleanup.use_masks})")
    visible_mask = filter_by_visibility(
        mesh.vertices,
        mesh,
        cameras,
        masks_dir,
        dilation_radius=config.cleanup.dilation_radius,
        show_progress=True,
        use_masks=config.cleanup.use_masks,
    )

    keep_mask = visible_mask
    n_visible = int(np.sum(visible_mask))

    # Robin's watertight culling: additionally drop visible, front-facing vertices that
    # project into the per-view hole-region masks (surfaces filling the GT's non-watertight
    # holes). Gated on cleanup.watertight_culling AND the masks dir existing (fail-open to
    # a plain clean if the masks are absent, but log it loudly).
    if config.cleanup.watertight_culling:
        issue_dir = config.get_masks_issue_watertight_dir(object_name)
        if issue_dir.exists() and any(issue_dir.glob("*.png")):
            print(f"  Watertight culling using: {issue_dir}")
            issue_counts = filter_by_issue_watertight(
                mesh.vertices,
                mesh,
                cameras,
                issue_dir,
                visible_mask=visible_mask,
                orientation_ratio_threshold=config.cleanup.watertight_orientation_ratio,
                show_progress=True,
            )
            keep_mask = visible_mask & (issue_counts == 0)
            n_culled = n_visible - int(np.sum(keep_mask))
            pct = (100.0 * n_culled / n_visible) if n_visible else 0.0
            print(f"  Watertight culling removed {n_culled} extra vertices "
                  f"({pct:.2f}% of the {n_visible} visible)")
        else:
            print(f"  WARNING: watertight_culling=True but no issue masks at {issue_dir} - skipping cull")

    n_kept = int(np.sum(keep_mask))
    n_removed = len(mesh.vertices) - n_kept
    print(f"  Visible vertices: {n_visible} ({100*n_visible/len(mesh.vertices):.1f}%)")
    print(f"  Kept after culling: {n_kept}  |  Removed total: {n_removed}")

    cleaned_mesh = filter_mesh_by_vertex_mask(mesh, keep_mask)
    print(f"  Cleaned mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.faces)} faces")

    cleaned_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    save_mesh(cleaned_mesh, cleaned_mesh_path)
    print(f"  Saved: {cleaned_mesh_path}")
