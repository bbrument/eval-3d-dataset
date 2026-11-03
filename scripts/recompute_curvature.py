#!/usr/bin/env python3
"""Recompute curvature_values.npy (VCM) with a specified ball radius r and tube thickness R.

Loads existing gt_pcd.npy and the raw GT mesh, computes curvature on its proxy tile by tile (src/core/proxy.py), saves.
Does NOT recompute gt_pcd or visibility.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from core.mesh import load_mesh
sys.path.append(str(Path(__file__).parent.parent))  # after src/: the repo root has a config/ folder
from src.pipeline.preprocess_gt import compute_gt_curvature  # relative imports: load through the src package
from config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--object", "-o", required=True)
    parser.add_argument("--radius", "-r", type=float, default=1.0, help="VCM ball radius r (mm)")
    parser.add_argument("--tube", type=float, default=0.5, help="VCM tube thickness R (mm)")
    parser.add_argument("--max-edge", type=float, default=0.15, help="longest edge of the proxy mesh (mm)")
    args = parser.parse_args()

    config = load_config(args.config)
    gt_dir = config.get_gt_dir(args.object)
    gt_pcd_path = gt_dir / "gt_pcd.npy"
    attributes_dir = gt_dir / "attributes"
    curvature_path = attributes_dir / "curvature_values.npy"

    if not gt_pcd_path.exists():
        print(f"ERROR: {gt_pcd_path} not found")
        sys.exit(1)

    # raw mesh: the visibility cleaning cuts holes whose rims the VCM would see as edges
    gt_mesh_path = config.get_gt_mesh_path(args.object, cleaned=False)
    if not gt_mesh_path.exists():
        print(f"ERROR: no GT mesh found for {args.object}")
        sys.exit(1)

    print(f"Object: {args.object}")
    print(f"Radius: {args.radius} mm, tube: {args.tube} mm")

    mesh = load_mesh(gt_mesh_path)
    print(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    gt_pcd = np.load(gt_pcd_path)
    print(f"GT pcd: {len(gt_pcd):,} points")

    print(f"Computing curvature on the proxy mesh (max edge {args.max_edge} mm)...")
    curvature = compute_gt_curvature(mesh, gt_pcd, radius=args.radius, tube=args.tube, max_edge=args.max_edge)

    attributes_dir.mkdir(parents=True, exist_ok=True)
    np.save(curvature_path, curvature)

    absc = np.abs(curvature)
    print(f"Saved: {curvature_path}")
    print(f"Stats: p50={np.percentile(absc,50):.4f} p90={np.percentile(absc,90):.4f} p99={np.percentile(absc,99):.4f} max={absc.max():.4f}")


if __name__ == "__main__":
    main()
