"""Tests for core/metrics.py - distance and F-score computation."""

import math

import numpy as np
import pytest

from src.core.metrics import (
    compute_distances,
    compute_chamfer,
    compute_fscore_curve,
    compute_metrics,
)


class TestComputeDistances:
    """Tests for compute_distances function."""

    def test_identical_clouds(self):
        """Distance should be zero for identical point clouds."""
        pcd = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        distances, indices = compute_distances(pcd, pcd)

        assert distances.shape == (3,)
        assert indices.shape == (3,)
        np.testing.assert_array_almost_equal(distances, [0, 0, 0])

    def test_known_distance(self):
        """Test with known distances."""
        pcd_a = np.array([[0, 0, 0]], dtype=np.float32)
        pcd_b = np.array([[1, 0, 0], [0, 2, 0]], dtype=np.float32)

        distances, indices = compute_distances(pcd_a, pcd_b)

        assert distances.shape == (1,)
        np.testing.assert_almost_equal(distances[0], 1.0)
        assert indices[0] == 0  # Nearest is first point at (1,0,0)

    def test_multiple_points(self):
        """Test bidirectional distance computation."""
        pcd_a = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
        pcd_b = np.array([[1, 0, 0]], dtype=np.float32)

        dist_a2b, idx_a2b = compute_distances(pcd_a, pcd_b)
        dist_b2a, idx_b2a = compute_distances(pcd_b, pcd_a)

        np.testing.assert_array_almost_equal(dist_a2b, [1.0, 1.0])
        np.testing.assert_array_almost_equal(dist_b2a, [1.0])


class TestComputeChamfer:
    """Tests for compute_chamfer function."""

    def test_zero_chamfer(self):
        """Chamfer distance should be zero for identical clouds."""
        dist_a2b = np.array([0, 0, 0], dtype=np.float32)
        dist_b2a = np.array([0, 0, 0], dtype=np.float32)

        result = compute_chamfer(dist_a2b, dist_b2a)

        assert result["chamfer"] == 0.0
        assert result["chamfer_a2b"] == 0.0
        assert result["chamfer_b2a"] == 0.0

    def test_known_chamfer(self):
        """Test Chamfer computation with known values."""
        dist_a2b = np.array([1, 2, 3], dtype=np.float32)  # mean = 2
        dist_b2a = np.array([2, 4], dtype=np.float32)  # mean = 3

        result = compute_chamfer(dist_a2b, dist_b2a)

        assert result["chamfer_a2b"] == pytest.approx(2.0)
        assert result["chamfer_b2a"] == pytest.approx(3.0)
        assert result["chamfer"] == pytest.approx(2.5)

    def test_max_dist_filtering(self):
        """Test outlier filtering with max_dist."""
        dist_a2b = np.array([1, 2, 100], dtype=np.float32)  # 100 is outlier
        dist_b2a = np.array([1, 1, 1], dtype=np.float32)

        result = compute_chamfer(dist_a2b, dist_b2a, max_dist=10.0)

        assert result["chamfer_a2b"] == pytest.approx(1.5)  # mean of [1, 2]
        assert result["n_filtered_a2b"] == 1
        assert result["n_filtered_b2a"] == 0

    def test_empty_after_filtering(self):
        """An empty filtered set must yield NaN, never 0.0.

        Regression test: 0.0 is the *best possible* score for a lower-is-better
        metric, so a reconstruction so misaligned that no point survives max_dist
        used to be recorded as perfect.
        """
        dist_a2b = np.array([100, 200], dtype=np.float32)
        dist_b2a = np.array([1], dtype=np.float32)

        result = compute_chamfer(dist_a2b, dist_b2a, max_dist=10.0)

        assert math.isnan(result["chamfer_a2b"])
        assert math.isnan(result["chamfer"]), "NaN must propagate to the combined chamfer"
        assert result["n_filtered_a2b"] == 2
        assert result["coverage_a2b"] == 0.0
        assert result["coverage"] == 0.0

    def test_coverage_reports_contributing_fraction(self):
        """coverage exposes the continuous bias, not just its extreme case."""
        dist_a2b = np.array([1, 2, 100, 200], dtype=np.float32)  # 2/4 survive
        dist_b2a = np.array([1, 1, 1, 100], dtype=np.float32)  # 3/4 survive

        result = compute_chamfer(dist_a2b, dist_b2a, max_dist=10.0)

        assert result["coverage_a2b"] == pytest.approx(0.5)
        assert result["coverage_b2a"] == pytest.approx(0.75)
        assert result["coverage"] == pytest.approx(0.5), "coverage is the worst direction"
        assert result["n_kept_a2b"] == 2
        assert result["chamfer_a2b"] == pytest.approx(1.5)

    def test_full_coverage_when_no_max_dist(self):
        """Without max_dist every point contributes."""
        dist_a2b = np.array([1, 2, 100], dtype=np.float32)
        dist_b2a = np.array([1, 2], dtype=np.float32)

        result = compute_chamfer(dist_a2b, dist_b2a, max_dist=None)

        assert result["coverage"] == pytest.approx(1.0)
        assert not math.isnan(result["chamfer"])

    def test_fscore_nan_when_no_points_at_all(self):
        """F-score curve must be NaN, not 0.0, when a side has no points."""
        dist_a2b = np.array([], dtype=np.float32)
        dist_b2a = np.array([300], dtype=np.float32)

        result = compute_fscore_curve(dist_a2b, dist_b2a, [0.5, 1.0])

        assert np.all(np.isnan(result["fscore"]))
        assert np.all(np.isnan(result["precision"]))

    def test_fscore_zero_is_still_legitimate(self):
        """Points that exist but fall beyond t give a real 0.0, not NaN."""
        dist_a2b = np.array([5.0], dtype=np.float32)
        dist_b2a = np.array([5.0], dtype=np.float32)

        result = compute_fscore_curve(dist_a2b, dist_b2a, [1.0])

        assert result["fscore"][0] == 0.0
        assert not np.isnan(result["fscore"][0])


class TestComputeFscoreCurve:
    """Tests for compute_fscore_curve function."""

    def test_perfect_score(self):
        """F-score should be 1.0 when all points are within threshold."""
        dist_a2b = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        dist_b2a = np.array([0.1, 0.2], dtype=np.float32)

        result = compute_fscore_curve(dist_a2b, dist_b2a, thresholds=[1.0])

        assert result["precision"][0] == pytest.approx(1.0)
        assert result["recall"][0] == pytest.approx(1.0)
        assert result["fscore"][0] == pytest.approx(1.0)

    def test_zero_score(self):
        """F-score should be 0 when no points are within threshold."""
        dist_a2b = np.array([10, 20, 30], dtype=np.float32)
        dist_b2a = np.array([10, 20], dtype=np.float32)

        result = compute_fscore_curve(dist_a2b, dist_b2a, thresholds=[1.0])

        assert result["precision"][0] == 0.0
        assert result["recall"][0] == 0.0
        assert result["fscore"][0] == 0.0

    def test_multiple_thresholds(self):
        """Test F-score curve with multiple thresholds."""
        # dist_a2b: distances from data (a) to GT (b) -> precision
        # dist_b2a: distances from GT (b) to data (a) -> recall
        dist_a2b = np.array([0.5, 1.5, 2.5], dtype=np.float32)
        dist_b2a = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)

        result = compute_fscore_curve(dist_a2b, dist_b2a, thresholds=[1.0, 2.0, 3.0])

        # F3 Fix: Clarify that comparison is strict (<), so value == threshold is NOT counted
        # At t=1.0:
        #   precision = (dist_a2b < 1.0).sum() / 3 = 1/3 (only 0.5 < 1.0; 1.5, 2.5 excluded)
        #   recall = (dist_b2a < 1.0).sum() / 4 = 1/4 (only 0.5 < 1.0; 1.0 itself is NOT < 1.0)
        assert result["precision"][0] == pytest.approx(1 / 3)
        assert result["recall"][0] == pytest.approx(1 / 4)

        # At t=2.0:
        #   precision = (dist_a2b < 2.0).sum() / 3 = 2/3 (0.5, 1.5 < 2.0)
        #   recall = (dist_b2a < 2.0).sum() / 4 = 3/4 (0.5, 1.0, 1.5 < 2.0)
        assert result["precision"][1] == pytest.approx(2 / 3)
        assert result["recall"][1] == pytest.approx(3 / 4)

        # At t=3.0: all within threshold
        assert result["precision"][2] == pytest.approx(1.0)
        assert result["recall"][2] == pytest.approx(1.0)

    def test_outliers_stay_in_the_denominator(self):
        """Regression: far points must count as misses, never be filtered away.

        The curve used to drop distances above `evaluation.max_dist` before taking
        `n_a` / `n_b`, which shrank the denominator without touching the numerator
        and reported `precision_true / coverage`. Here 2 of 4 reconstructed points
        and 1 of 4 GT points are gross outliers: precision at t=1.0 is 2/4, not 2/2.
        """
        dist_a2b = np.array([0.5, 0.5, 100.0, 200.0], dtype=np.float32)
        dist_b2a = np.array([0.5, 0.5, 0.5, 300.0], dtype=np.float32)

        result = compute_fscore_curve(dist_a2b, dist_b2a, thresholds=[1.0])

        assert result["precision"][0] == pytest.approx(0.5)
        assert result["recall"][0] == pytest.approx(0.75)
        assert result["fscore"][0] == pytest.approx(2 * 0.5 * 0.75 / 1.25)

    def test_threshold_above_outlier_range_is_not_saturated(self):
        """Regression: a threshold at or beyond the old max_dist forced 1.0.

        Every surviving point satisfied `d < max_dist <= t`, so precision, recall
        and F-score were pinned to 1.0 by construction — the case actually hit by
        the shipped configs (martine: max_dist 4.0 with a 5.0 threshold; sk3d:
        max_dist 5.0 with a 5.0 threshold).
        """
        dist_a2b = np.array([1.0, 1.0, 1.0, 100.0], dtype=np.float32)
        dist_b2a = np.array([1.0, 100.0], dtype=np.float32)

        result = compute_fscore_curve(dist_a2b, dist_b2a, thresholds=[5.0])

        assert result["precision"][0] == pytest.approx(0.75)
        assert result["recall"][0] == pytest.approx(0.5)


class TestComputeMetrics:
    """Integration tests for compute_metrics."""

    def test_combines_chamfer_and_fscore(self):
        """compute_metrics should return both Chamfer and F-score results."""
        dist_data2gt = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        dist_gt2data = np.array([0.5, 0.5], dtype=np.float32)

        result = compute_metrics(dist_data2gt, dist_gt2data, thresholds=[1.0, 2.0])

        # Check Chamfer fields exist
        assert "chamfer" in result
        assert "chamfer_a2b" in result
        assert "chamfer_b2a" in result

        # Check F-score fields exist
        assert "thresholds" in result
        assert "precision" in result
        assert "recall" in result
        assert "fscore" in result

        # Check values
        assert result["chamfer_a2b"] == pytest.approx(1.0)
        assert result["chamfer_b2a"] == pytest.approx(0.5)

    # F5 Fix: Verify numpy arrays are converted to lists
    def test_returns_lists_not_numpy_arrays(self):
        """compute_metrics should convert numpy arrays to lists for JSON serialization."""
        dist_data2gt = np.array([0.5, 1.0], dtype=np.float32)
        dist_gt2data = np.array([0.5], dtype=np.float32)

        result = compute_metrics(dist_data2gt, dist_gt2data, thresholds=[1.0, 2.0])

        # These fields should be lists, not numpy arrays
        assert isinstance(result["thresholds"], list)
        assert isinstance(result["precision"], list)
        assert isinstance(result["recall"], list)
        assert isinstance(result["fscore"], list)

        # Chamfer values should be floats, not numpy scalars
        assert isinstance(result["chamfer"], float)
        assert isinstance(result["chamfer_a2b"], float)
        assert isinstance(result["chamfer_b2a"], float)

    def test_max_dist_clips_chamfer_only(self):
        """max_dist bounds the Chamfer average but must not touch precision/recall."""
        dist_data2gt = np.array([1.0, 1.0, 100.0], dtype=np.float32)
        dist_gt2data = np.array([1.0, 1.0], dtype=np.float32)

        result = compute_metrics(
            dist_data2gt, dist_gt2data, thresholds=[2.0], max_dist=10.0
        )

        # Chamfer: outlier dropped, averaged over the 2 survivors
        assert result["chamfer_a2b"] == pytest.approx(1.0)
        assert result["n_filtered_a2b"] == 1
        assert result["coverage_a2b"] == pytest.approx(2 / 3)

        # Precision: the outlier is a miss, so 2/3 — not 2/2
        assert result["precision"][0] == pytest.approx(2 / 3)
        assert result["recall"][0] == pytest.approx(1.0)
