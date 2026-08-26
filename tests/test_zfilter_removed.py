"""Regression tests: the relative z-cut (z > min(z_GT)) is no longer applied.

Historically ``evaluate`` dropped every reconstructed point whose z was at or
below the ground-truth floor. That height cut has been removed: only NaN points
are discarded (numerical hygiene). These tests pin the new behaviour so it
cannot silently regress:

  * points below ``min(z_GT)`` are kept and contribute to the distances,
  * NaN points are still removed,
  * finite points are preserved untouched.
"""

import numpy as np

from src.core.sampling import remove_nan_points
from src.core.metrics import compute_distances


def test_points_below_gt_floor_are_kept():
    """A reconstructed point well below the GT floor must survive filtering."""
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gt_min_z = gt[:, 2].min()  # 0.0

    data = np.array([
        [0.0, 0.0, -5.0],   # far below the GT floor -> previously removed
        [0.5, 0.5, 0.0],
    ])

    filtered = remove_nan_points(data)

    # No height cut: nothing is dropped for being below the floor.
    assert len(filtered) == 2
    assert np.any(filtered[:, 2] < gt_min_z)


def test_subfloor_point_contributes_to_distances():
    """The kept sub-floor point must show up in the reconstructed->GT distances."""
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    data = np.array([[0.0, 0.0, -5.0], [0.5, 0.5, 0.0]])

    filtered = remove_nan_points(data)
    dist_data2gt, _ = compute_distances(filtered, gt)

    # The sub-floor point is 5 units from the nearest GT point; if it had been
    # cut, this large distance would be absent.
    assert len(dist_data2gt) == 2
    assert dist_data2gt.max() >= 5.0 - 1e-6


def test_nan_points_are_removed():
    """Rows containing any NaN coordinate are dropped."""
    data = np.array([
        [0.0, 0.0, 0.0],
        [np.nan, 1.0, 2.0],
        [3.0, np.nan, 4.0],
        [5.0, 6.0, np.nan],
    ])

    filtered = remove_nan_points(data)

    assert len(filtered) == 1
    assert np.isfinite(filtered).all()
    np.testing.assert_array_equal(filtered[0], np.array([0.0, 0.0, 0.0]))


def test_finite_points_preserved_in_order():
    """Finite points pass through unchanged, order preserved."""
    data = np.array([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0], [7.0, 8.0, 9.0]])

    filtered = remove_nan_points(data)

    np.testing.assert_array_equal(filtered, data)
