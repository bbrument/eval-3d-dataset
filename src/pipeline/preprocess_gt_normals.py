"""Preprocess GT normals: render normal maps from GT mesh for all views."""

from pathlib import Path

from ..config import Config
from ..core.mesh import load_mesh
from ..core.normals import render_normals_from_mesh


def preprocess_gt_normals(
    config: Config,
    object_name: str,
    pose_sources: list[str] | None = None,
    downscales: list[str] | None = None,
    force: bool = False,
) -> None:
    """Render GT normal maps for an object across pose sources and downscales.

    Args:
        config: Pipeline configuration (must have normals section).
        object_name: Object name.
        pose_sources: List of pose sources to render (None = all configured).
        downscales: List of downscales (None = ["d1", "d2", "d4", "d8"]).
        force: Overwrite existing files.
    """
    if config.normals is None:
        raise ValueError("normals config section is required for preprocess-gt-normals")

    all_downscales = downscales or ["d1", "d2", "d4", "d8"]
    all_pose_sources = pose_sources or list(config.normals.pose_sources.keys())

    gt_mesh_path = config.get_gt_mesh_path(object_name)
    if config.normals.gt_mesh_path:
        gt_mesh_path = Path(config.normals.gt_mesh_path)

    if not gt_mesh_path.exists():
        raise FileNotFoundError(f"GT mesh not found: {gt_mesh_path}")

    print(f"Loading GT mesh: {gt_mesh_path}")
    mesh = load_mesh(gt_mesh_path)
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    for pose_source in all_pose_sources:
        for downscale in all_downscales:
            sfm_path = config.resolve_normals_sfm_path(object_name, pose_source, downscale)

            if not sfm_path.exists():
                print(f"  Skipping {pose_source}/{downscale}: {sfm_path} not found")
                continue

            output_dir = config.get_gt_normals_dir(object_name, pose_source, downscale)

            # Check if already done
            normals_dir = output_dir / "normals"
            if normals_dir.exists() and not force:
                existing = list(normals_dir.glob("*.png"))
                if existing:
                    print(f"  {pose_source}/{downscale}: {len(existing)} normals exist, skipping")
                    continue

            print(f"  Rendering GT normals: {pose_source}/{downscale}")
            print(f"    SfM: {sfm_path}")
            print(f"    Output: {output_dir}")

            rendered = render_normals_from_mesh(
                mesh,
                sfm_path,
                output_dir,
                chunk_size=config.normals.rendering.chunk_size,
                samples=config.normals.rendering.samples,
                force=force,
            )
            print(f"    Rendered {len(rendered)} normal maps")
