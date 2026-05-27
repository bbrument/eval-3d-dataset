"""Tests for core/sampling.py - mesh upsampling and downsampling."""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors  # F7 Fix: Move import to top

from src.core.sampling import sample_single_tri, upsample_mesh, downsample_pcd


class TestSampleSingleTri:
    """Tests for sample_single_tri function."""

    def test_no_samples(self):
        """Test with n1=0, n2=0 returns empty array (no interior points)."""
        v1 = np.array([[1, 0, 0]], dtype=np.float32)
        v2 = np.array([[0, 1, 0]], dtype=np.float32)
        tri_vert = np.array([[0, 0, 0]], dtype=np.float32)

        result = sample_single_tri((0, 0, v1, v2, tri_vert))

        # F11 Fix: Correct understanding of the algorithm
        # With n=0, grid is 1x1 with center (0.5, 0.5)
        # Barycentric sum = 0.5 + 0.5 = 1.0, which is NOT < 1
        # So we get zero interior points (empty array)
        assert result.shape == (0, 3)  # Empty array with 3 columns

    def test_samples_inside_triangle(self):
        """Sampled points should be inside the triangle."""
        v1 = np.array([[1, 0, 0]], dtype=np.float32)
        v2 = np.array([[0, 1, 0]], dtype=np.float32)
        tri_vert = np.array([[0, 0, 0]], dtype=np.float32)

        result = sample_single_tri((3, 3, v1, v2, tri_vert))

        # All z coordinates should be 0
        np.testing.assert_array_almost_equal(result[:, 2], 0)

        # All points should have x + y < 1 (inside triangle)
        sums = result[:, 0] + result[:, 1]
        assert np.all(sums < 1.0 + 1e-6)

    def test_returns_3d_points(self):
        """Result should have shape (N, 3)."""
        v1 = np.array([[2, 0, 0]], dtype=np.float32)
        v2 = np.array([[0, 2, 0]], dtype=np.float32)
        tri_vert = np.array([[1, 1, 1]], dtype=np.float32)

        result = sample_single_tri((2, 2, v1, v2, tri_vert))

        assert result.ndim == 2
        assert result.shape[1] == 3


class TestUpsampleMesh:
    """Tests for upsample_mesh function."""

    def test_includes_original_vertices(self):
        """Upsampled cloud should include original vertices."""
        # Simple triangle
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]])

        result = upsample_mesh(vertices, faces, density=0.5, n_jobs=1)

        # Original vertices should be in result
        assert len(result) >= 3

        # Check first 3 points are original vertices
        np.testing.assert_array_almost_equal(result[:3], vertices)

    def test_higher_density_more_points(self):
        """Higher density (smaller spacing) should produce more points."""
        vertices = np.array(
            [[0, 0, 0], [10, 0, 0], [5, 10, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2]])

        result_low = upsample_mesh(vertices, faces, density=2.0, n_jobs=1)
        result_high = upsample_mesh(vertices, faces, density=0.5, n_jobs=1)

        assert len(result_high) > len(result_low)

    def test_empty_faces(self):
        """Handle mesh with no faces."""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int64)

        result = upsample_mesh(vertices, faces, density=0.1, n_jobs=1)

        # Should return original vertices
        np.testing.assert_array_equal(result, vertices)

    def test_degenerate_triangle(self):
        """Handle degenerate triangles (zero area)."""
        # Collinear points form degenerate triangle
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [0, 1, 3]])  # First is degenerate

        result = upsample_mesh(vertices, faces, density=0.5, n_jobs=1)

        # F9 Fix: Verify degenerate handling more thoroughly
        # Original vertices should be included
        assert len(result) >= 4

        # Compare with non-degenerate only mesh
        vertices_valid = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
        )
        faces_valid = np.array([[0, 1, 2]])
        result_valid = upsample_mesh(vertices_valid, faces_valid, density=0.5, n_jobs=1)

        # Result with degenerate should have same sampled points from valid triangle
        # plus the extra vertex (2,0,0), but no extra samples from degenerate
        # The degenerate triangle should contribute no sampled points
        assert len(result) == len(result_valid) + 1  # +1 for vertex (2,0,0)


class TestDownsamplePcd:
    """Tests for downsample_pcd function."""

    def test_reduces_point_count(self):
        """Downsampling should reduce number of points."""
        # Create dense grid
        x, y, z = np.mgrid[0:10:0.1, 0:10:0.1, 0:1:1]
        pcd = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T.astype(np.float32)

        result = downsample_pcd(pcd, density=1.0)

        assert len(result) < len(pcd)

    def test_minimum_spacing(self):
        """Points should be at least density apart after downsampling."""
        np.random.seed(42)
        pcd = np.random.rand(1000, 3).astype(np.float32) * 10

        density = 0.5
        result = downsample_pcd(pcd, density=density, seed=42)

        # Check minimum distance between any two points
        if len(result) > 1:
            # F7 Fix: Use top-level import
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(result)
            distances, _ = nn.kneighbors(result)
            min_dist = distances[:, 1].min()  # Second neighbor (first is self)

            # Should be approximately density (allowing some tolerance)
            assert min_dist >= density * 0.9

    def test_reproducibility_with_seed(self):
        """Same seed should produce same result."""
        pcd = np.random.rand(100, 3).astype(np.float32)

        result1 = downsample_pcd(pcd, density=0.1, seed=42)
        result2 = downsample_pcd(pcd, density=0.1, seed=42)

        np.testing.assert_array_equal(result1, result2)

    def test_no_shuffle(self):
        """Without shuffle, order should be preserved."""
        pcd = np.array(
            [[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=np.float32
        )

        result = downsample_pcd(pcd, density=0.3, shuffle=False)

        # First point should be kept
        np.testing.assert_array_equal(result[0], [0, 0, 0])
