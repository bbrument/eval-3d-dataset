"""F8 Fix: Integration tests for pipeline stages."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import json

from src.core.mesh import load_mesh, save_mesh, filter_mesh_by_vertex_mask
from src.core.metrics import compute_distances, compute_metrics
from src.core.sampling import upsample_mesh, downsample_pcd
from src.io.results import save_metrics, load_metrics, save_distances, load_distances
from src.io.masks import load_gt_pcd, load_visibility_count, load_curvature_values


class TestMeshUtilities:
    """Integration tests for mesh.py utilities."""

    def test_filter_mesh_preserves_topology(self, tmp_path):
        """Filtering should preserve valid mesh topology."""
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        # Create simple mesh (tetrahedron)
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [0.5, 0.5, 1]
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [0, 2, 3]
        ])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Filter out one vertex
        mask = np.array([True, True, True, False])  # Remove top vertex
        filtered = filter_mesh_by_vertex_mask(mesh, mask)

        # Should have 3 vertices and 1 face (bottom triangle)
        assert len(filtered.vertices) == 3
        assert len(filtered.faces) == 1


class TestResultsIO:
    """Integration tests for results I/O."""

    def test_save_and_load_metrics(self, tmp_path):
        """Metrics should round-trip through save/load."""
        metrics = {
            "chamfer": 0.5,
            "chamfer_a2b": 0.4,
            "chamfer_b2a": 0.6,
            "fscore_1.0": 0.85,
            "n_gt_points": 1000,
            "n_data_points": 950
        }

        save_metrics(tmp_path, metrics)
        loaded = load_metrics(tmp_path)

        assert loaded["chamfer"] == pytest.approx(0.5)
        assert loaded["fscore_1.0"] == pytest.approx(0.85)

    def test_save_and_load_distances(self, tmp_path):
        """Distance arrays should round-trip through save/load."""
        gt2data_dist = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        gt2data_idx = np.array([0, 1, 2], dtype=np.int64)
        data2gt_dist = np.array([0.15, 0.25], dtype=np.float32)
        data2gt_idx = np.array([1, 2], dtype=np.int64)
        gt_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        data_points = np.array([[0.1, 0, 0], [0.9, 0, 0]], dtype=np.float32)

        save_distances(
            tmp_path,
            gt2data_dist=gt2data_dist,
            gt2data_idx=gt2data_idx,
            data2gt_dist=data2gt_dist,
            data2gt_idx=data2gt_idx,
            gt_points=gt_points,
            data_points=data_points
        )

        loaded = load_distances(tmp_path)

        np.testing.assert_array_almost_equal(loaded["gt2data_dist"], gt2data_dist)
        np.testing.assert_array_equal(loaded["gt2data_idx"], gt2data_idx)
        np.testing.assert_array_almost_equal(loaded["gt_points"], gt_points)


class TestEndToEndEvaluation:
    """End-to-end integration test for evaluation flow."""

    def test_full_evaluation_pipeline(self, tmp_path):
        """Test complete evaluation from mesh to metrics."""
        try:
            import trimesh
        except ImportError:
            pytest.skip("trimesh not installed")

        # Create two similar meshes (GT and reconstructed)
        # GT: unit cube
        gt_vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float64)
        gt_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5],  # right
        ])

        # Reconstructed: slightly smaller cube (90% scale)
        data_vertices = gt_vertices * 0.9 + 0.05
        data_faces = gt_faces.copy()

        # Step 1: Upsample meshes to point clouds
        gt_pcd = upsample_mesh(gt_vertices, gt_faces, density=0.2, n_jobs=1)
        data_pcd = upsample_mesh(data_vertices, data_faces, density=0.2, n_jobs=1)

        # Step 2: Downsample for uniform density
        gt_pcd_down = downsample_pcd(gt_pcd, density=0.1, seed=42)
        data_pcd_down = downsample_pcd(data_pcd, density=0.1, seed=42)

        # Step 3: Compute distances
        dist_gt2data, idx_gt2data = compute_distances(gt_pcd_down, data_pcd_down)
        dist_data2gt, idx_data2gt = compute_distances(data_pcd_down, gt_pcd_down)

        # Step 4: Compute metrics
        metrics = compute_metrics(
            dist_data2gt, dist_gt2data,
            thresholds=[0.05, 0.1, 0.2]
        )

        # Verify metrics make sense
        assert metrics["chamfer"] > 0  # Not identical meshes
        assert metrics["chamfer"] < 0.5  # But similar
        assert len(metrics["fscore"]) == 3  # Three thresholds

        # Step 5: Save and reload
        save_metrics(tmp_path, metrics)
        save_distances(
            tmp_path,
            gt2data_dist=dist_gt2data,
            gt2data_idx=idx_gt2data,
            data2gt_dist=dist_data2gt,
            data2gt_idx=idx_data2gt,
            gt_points=gt_pcd_down,
            data_points=data_pcd_down
        )

        # Verify files exist
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "distances" / "gt2data_dist.npy").exists()

        # Verify reload works
        loaded_metrics = load_metrics(tmp_path)
        assert loaded_metrics["chamfer"] == pytest.approx(metrics["chamfer"])


class TestMasksIO:
    """Integration tests for masks I/O."""

    def test_load_gt_attributes(self, tmp_path):
        """Test loading GT point cloud and attributes."""
        # Create mock GT directory structure
        gt_dir = tmp_path / "Groundtruth"
        gt_dir.mkdir()
        attr_dir = gt_dir / "attributes"
        attr_dir.mkdir()

        # Create mock data
        gt_pcd = np.random.rand(100, 3).astype(np.float32)
        visibility = np.random.randint(1, 85, size=100).astype(np.int32)
        curvature = np.random.rand(100).astype(np.float32) * 2 - 1  # -1 to 1

        np.save(gt_dir / "gt_pcd.npy", gt_pcd)
        np.save(attr_dir / "visibility_count.npy", visibility)
        np.save(attr_dir / "curvature_values.npy", curvature)

        # Load and verify
        loaded_pcd = load_gt_pcd(gt_dir)
        loaded_vis = load_visibility_count(gt_dir)
        loaded_curv = load_curvature_values(gt_dir)

        np.testing.assert_array_equal(loaded_pcd, gt_pcd)
        np.testing.assert_array_equal(loaded_vis, visibility)
        np.testing.assert_array_almost_equal(loaded_curv, curvature)
