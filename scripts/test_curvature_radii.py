#!/usr/bin/env python3
"""Test curvature computation with different radii on a decimated mesh.

Generates side-by-side visualization renders for comparison.
"""

import sys
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from pathlib import Path
from PIL import Image
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.mesh import load_mesh
from core.colormap import color_by_curvature
from core.rendering import render_views, PYRENDER_AVAILABLE
from core.camera import load_cameras_auto
from config import load_config


def compute_curvature_fast(mesh, radius):
    """Compute curvature with given radius, with progress reporting."""
    n = len(mesh.vertices)
    tree = cKDTree(mesh.vertices)
    neighbors_list = tree.query_ball_point(mesh.vertices, r=radius, workers=-1)

    avg_neighbors = np.mean([len(x) for x in neighbors_list])
    print(f"    Avg neighbors per vertex: {avg_neighbors:.0f}")

    curvature = np.zeros(n, dtype=np.float32)
    t0 = time.time()

    for i in range(n):
        idx = neighbors_list[i]
        if len(idx) < 6:
            continue

        pts = mesh.vertices[idx]
        center = mesh.vertices[i]
        pts_centered = pts - center
        normal = mesh.vertex_normals[i]

        if abs(normal[0]) < 0.9:
            tangent1 = np.cross(normal, [1, 0, 0])
        else:
            tangent1 = np.cross(normal, [0, 1, 0])
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)

        u = pts_centered @ tangent1
        v = pts_centered @ tangent2
        w = pts_centered @ normal

        A = np.column_stack([u**2, u*v, v**2])
        try:
            params, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
            a, b, c = params
            tr = 2*a + 2*c
            det = 4*a*c - b**2
            disc = max(0, tr**2 - 4*det)
            k1 = (tr + np.sqrt(disc)) / 2
            k2 = (tr - np.sqrt(disc)) / 2
            curvature[i] = k1 if abs(k1) > abs(k2) else k2
        except np.linalg.LinAlgError:
            continue

        if i > 0 and i % 100000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n - i) / rate
            print(f"    {i}/{n} ({100*i/n:.1f}%) - {rate:.0f} v/s - ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    return curvature


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", "-o", default="11_ecorce")
    parser.add_argument("--method", "-m", default="colmap_d2_refin")
    parser.add_argument("--view", "-v", type=int, nargs="+", default=[0, 40])
    args = parser.parse_args()

    config_path = Path("/home/babrument/dev/eval_dataset/eval_pipeline/config/martine_watcher.yaml")
    config = load_config(str(config_path))

    object_name = args.object
    method_name = args.method

    radii_mm = [0.5, 1.0, 2.0, 4.0, 8.0]
    decimation_target = 500_000
    view_indices = args.view

    gt_dir = config.get_gt_dir(object_name)
    out_base = gt_dir / "curvature_tests"
    out_base.mkdir(parents=True, exist_ok=True)

    # Load GT mesh
    gt_mesh_path = config.get_gt_mesh_path(object_name, cleaned=False)
    print(f"Loading GT mesh: {gt_mesh_path}")
    gt_mesh_full = load_mesh(gt_mesh_path)
    print(f"  Full mesh: {len(gt_mesh_full.vertices):,} vertices, {len(gt_mesh_full.faces):,} faces")

    # Decimate for faster computation
    if len(gt_mesh_full.faces) > decimation_target:
        print(f"  Decimating to {decimation_target} faces...")
        gt_mesh = gt_mesh_full.simplify_quadric_decimation(face_count=decimation_target)
        print(f"  Decimated: {len(gt_mesh.vertices):,} vertices, {len(gt_mesh.faces):,} faces")
    else:
        gt_mesh = gt_mesh_full

    # Load cameras
    cameras_path = config.get_cameras_path(object_name, method_name)
    cameras = load_cameras_auto(cameras_path)

    # Compute curvature for each radius
    for radius in radii_mm:
        tag = f"r{radius:.1f}mm"
        out_dir = out_base / tag

        # Check if already computed
        curv_path = out_dir / "curvature.npy"
        if curv_path.exists():
            print(f"\n=== Radius {radius}mm: loading cached ===")
            curvature = np.load(curv_path)
        else:
            print(f"\n=== Radius {radius}mm: computing ===")
            curvature = compute_curvature_fast(gt_mesh, radius)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(curv_path, curvature)

        abs_curv = np.abs(curvature)
        p50, p90, p99 = np.percentile(abs_curv, [50, 90, 99])
        print(f"  Stats: p50={p50:.4f} p90={p90:.4f} p99={p99:.4f}")

        # Render
        colored, cb_params = color_by_curvature(gt_mesh, curvature, cmap="coolwarm")

        render_dir = out_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        render_views(colored, cameras, view_indices, render_dir,
                     scale=2.0, crop=True, crop_margin=15)

        print(f"  Renders saved to {render_dir}")

    # Also render the current (full mesh) curvature for comparison
    existing_curv_path = gt_dir / "attributes" / "curvature_values.npy"
    if existing_curv_path.exists():
        print(f"\n=== Current (original) curvature ===")
        curv_orig = np.load(existing_curv_path)
        # Transfer to decimated mesh via KNN
        tree = cKDTree(gt_mesh_full.vertices)
        _, idx = tree.query(gt_mesh.vertices, k=1)
        curv_on_dec = curv_orig[idx] if len(curv_orig) == len(gt_mesh_full.vertices) else curv_orig

        out_dir = out_base / "current"
        out_dir.mkdir(parents=True, exist_ok=True)
        colored, _ = color_by_curvature(gt_mesh, curv_on_dec, cmap="coolwarm")
        render_dir = out_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        render_views(colored, cameras, view_indices, render_dir,
                     scale=2.0, crop=True, crop_margin=15)
        print(f"  Renders saved to {render_dir}")

    print(f"\nAll results in: {out_base}")
    print("Compare renders across radii to pick the best one.")


if __name__ == "__main__":
    main()
