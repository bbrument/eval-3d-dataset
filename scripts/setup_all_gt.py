#!/usr/bin/env python3
"""Setup GT directories and compute attributes for all objects."""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.camera import load_cameras_auto
from core.mesh import load_mesh, compute_vertex_curvature
from core.visibility import compute_visibility_count


def setup_gt_for_object(
    object_name: str,
    objects_root: Path,
    eval_root: Path,
    data_subdir: str = "09_unimsps_d2",
    gt_resolution: str = "10mm",
    force: bool = False,
):
    """Setup GT directory and compute attributes for one object.

    Args:
        object_name: Object name (e.g., "01_rock")
        objects_root: Root of objects directory
        eval_root: Root of eval directory
        data_subdir: Subdirectory containing cameras/masks
        gt_resolution: GT resolution to use (10mm, 15mm, 20mm)
        force: Overwrite existing files
    """
    print(f"\n{'='*60}")
    print(f"Processing: {object_name}")
    print(f"{'='*60}")

    # Paths
    obj_dir = objects_root / object_name
    gt_source_dir = obj_dir / "00_gt"
    data_dir = obj_dir / data_subdir
    eval_obj_dir = eval_root / object_name
    gt_eval_dir = eval_obj_dir / "Groundtruth"
    attributes_dir = gt_eval_dir / "attributes"

    # Check source GT exists
    gt_mesh_name = f"gt_{gt_resolution}.ply"
    gt_mesh_source = gt_source_dir / gt_mesh_name
    if not gt_mesh_source.exists():
        # Try alternative names
        for alt in ["gt_10mm.ply", "gt_15mm.ply", "gt_20mm.ply"]:
            alt_path = gt_source_dir / alt
            if alt_path.exists():
                gt_mesh_source = alt_path
                gt_mesh_name = alt
                break

    if not gt_mesh_source.exists():
        print(f"  ERROR: No GT mesh found in {gt_source_dir}")
        return False

    # Check cameras exist
    cameras_path = None
    for cam_file in ["sfm.json", "cameras.npz", "cameras.json"]:
        candidate = data_dir / cam_file
        if candidate.exists():
            cameras_path = candidate
            break

    if cameras_path is None:
        print(f"  ERROR: No camera file found in {data_dir}")
        return False

    # Check masks exist
    masks_dir = data_dir / "masks"
    if not masks_dir.exists():
        masks_dir = data_dir / "mask"
    if not masks_dir.exists():
        print(f"  ERROR: No masks directory found in {data_dir}")
        return False

    print(f"  GT mesh: {gt_mesh_source}")
    print(f"  Cameras: {cameras_path}")
    print(f"  Masks: {masks_dir}")

    # Create directories
    gt_eval_dir.mkdir(parents=True, exist_ok=True)
    attributes_dir.mkdir(parents=True, exist_ok=True)

    # Create symlink to GT mesh
    gt_link = gt_eval_dir / gt_mesh_name
    if gt_link.exists() or gt_link.is_symlink():
        if force:
            gt_link.unlink()
        else:
            print(f"  GT link already exists: {gt_link}")

    if not gt_link.exists():
        gt_link.symlink_to(gt_mesh_source)
        print(f"  Created symlink: {gt_link} -> {gt_mesh_source}")

    # Load mesh and cameras
    print(f"  Loading mesh...")
    mesh = load_mesh(gt_mesh_source)
    print(f"    Vertices: {len(mesh.vertices)}")

    print(f"  Loading cameras...")
    cameras = load_cameras_auto(cameras_path)
    print(f"    Views: {len(cameras['K'])}")

    # Compute visibility
    visibility_path = attributes_dir / "visibility_count.npy"
    if visibility_path.exists() and not force:
        print(f"  Visibility already computed: {visibility_path}")
    else:
        print(f"  Computing visibility...")
        visibility_count = compute_visibility_count(
            mesh.vertices,
            mesh,
            cameras,
            masks_dir,
            dilation_radius=12,
            show_progress=True,
        )
        np.save(visibility_path, visibility_count)
        print(f"    Saved: {visibility_path}")
        print(f"    Min/Max visibility: {visibility_count.min()}/{visibility_count.max()}")

    # Compute curvature
    curvature_path = attributes_dir / "curvature_values.npy"
    if curvature_path.exists() and not force:
        print(f"  Curvature already computed: {curvature_path}")
    else:
        print(f"  Computing curvature...")
        curvature = compute_vertex_curvature(mesh)
        np.save(curvature_path, curvature)
        print(f"    Saved: {curvature_path}")
        print(f"    Min/Max curvature: {curvature.min():.4f}/{curvature.max():.4f}")

    # Save GT point cloud if not exists
    gt_pcd_path = gt_eval_dir / "gt_pcd.npy"
    if not gt_pcd_path.exists() or force:
        np.save(gt_pcd_path, mesh.vertices.astype(np.float32))
        print(f"  Saved GT point cloud: {gt_pcd_path}")

    print(f"  Done: {object_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup GT for all objects")
    parser.add_argument(
        "--objects-root",
        type=Path,
        default=Path("/home/babrument/dev/alicevision/dataset_temp/objects"),
        help="Root directory containing object folders",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/home/babrument/dev/alicevision/dataset_temp/eval"),
        help="Root directory for eval output",
    )
    parser.add_argument(
        "--data-subdir",
        default="09_unimsps_d2",
        help="Subdirectory containing cameras/masks",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        help="Specific objects to process (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    args = parser.parse_args()

    # Get list of objects
    if args.objects:
        objects = args.objects
    else:
        objects = sorted([
            d.name for d in args.objects_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
            and d.name != "Calib"
        ])

    print(f"Processing {len(objects)} objects...")
    print(f"Objects root: {args.objects_root}")
    print(f"Eval root: {args.eval_root}")
    print(f"Data subdir: {args.data_subdir}")

    success = []
    failed = []

    for obj in objects:
        try:
            result = setup_gt_for_object(
                obj,
                args.objects_root,
                args.eval_root,
                args.data_subdir,
                force=args.force,
            )
            if result:
                success.append(obj)
            else:
                failed.append(obj)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(obj)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Success: {len(success)}/{len(objects)}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
