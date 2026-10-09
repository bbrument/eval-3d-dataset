"""Exact nearest-GT-point lookup that stays fast for vertices far from the GT.

Regression for 10_colander: method meshes span the bowl opening (a "lid" ~R away
from every GT point). scipy's cKDTree bounds its cells by split planes only, so an
exact query from there visits most of the 23 M GT points; the visualization took
4-6 h. The index must return exactly the cKDTree nearest neighbour, much faster.
"""

import time

import numpy as np
from scipy.spatial import cKDTree

from src.core.nearest import NearestPointIndex


def _perforated_bowl(n_points: int, seed: int = 0) -> np.ndarray:
    """Hemispherical shell (inner R=50, outer R=51 mm), opening at z=0, with holes."""
    rng = np.random.default_rng(seed)
    pts = []
    while sum(len(p) for p in pts) < n_points:
        d = rng.normal(size=(n_points, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d = d[d[:, 2] < 0]
        azimuth = np.arctan2(d[:, 1], d[:, 0])
        polar = np.arccos(-d[:, 2])
        hole = (np.mod(azimuth, 0.2) < 0.06) & (np.mod(polar, 0.2) < 0.06) & (polar > 0.3)
        d = d[~hole]
        radius = np.where(rng.random(len(d)) < 0.5, 50.0, 51.0)
        pts.append(d * radius[:, None] + rng.normal(scale=0.05, size=d.shape))
    return np.concatenate(pts)[:n_points]


def _queries(points: np.ndarray, n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    near = points[rng.choice(len(points), n, replace=False)] + rng.normal(scale=0.3, size=(n, 3))
    rho = 45 * np.sqrt(rng.random(n))
    phi = rng.random(n) * 2 * np.pi
    lid = np.c_[rho * np.cos(phi), rho * np.sin(phi), rng.uniform(-2, 8, n)]
    outside = rng.normal(size=(n, 3))
    outside = outside / np.linalg.norm(outside, axis=1, keepdims=True) * rng.uniform(55, 150, (n, 1))
    return np.concatenate([near, lid, outside])


def test_indices_equal_exact_ckdtree_near_and_far():
    points = _perforated_bowl(120_000)
    queries = _queries(points, 1500)
    expected = cKDTree(points).query(queries, k=1)[1]

    index = NearestPointIndex(points, near_radius=1.0, workers=2)
    actual = index.query(queries)

    np.testing.assert_array_equal(actual, expected)
    # Reusing the index (GT mesh then method mesh) gives the same answers.
    np.testing.assert_array_equal(index.query(queries[::-1]), expected[::-1])


def test_exact_for_any_near_radius_and_empty_input():
    points = _perforated_bowl(30_000, seed=3)
    queries = _queries(points, 300, seed=4)
    expected = cKDTree(points).query(queries, k=1)[1]
    for radius in (0.0, 0.2, 5.0, np.inf):
        np.testing.assert_array_equal(NearestPointIndex(points, near_radius=radius).query(queries), expected)
    assert NearestPointIndex(points).query(np.empty((0, 3))).shape == (0,)


def test_exact_ties_resolved_like_default_ckdtree():
    """Equidistant points must be broken exactly as cKDTree(points) (leafsize 16) does.

    The historical renders used ``cKDTree(gt_pcd).query``; real GT kits contain exact
    ties (19_straw_bob: 147 decimated-GT vertices, one of them flips its label).
    """
    g = np.arange(-20, 21, dtype=np.float64)
    x, y, z = np.meshgrid(g, g, g, indexing="ij")
    cube = np.c_[x.ravel(), y.ravel(), z.ravel()]
    points = cube[np.max(np.abs(cube), axis=1) == 20]  # integer lattice on a cube surface
    rng = np.random.default_rng(7)
    near = points[rng.choice(len(points), 1500)] + rng.choice([-0.5, 0.0, 0.5], size=(1500, 3))
    far = rng.integers(-12, 13, size=(1500, 3)) + rng.choice([0.0, 0.5], size=(1500, 3))
    queries = np.concatenate([near, far])
    expected = cKDTree(points).query(queries, k=1)[1]

    actual = NearestPointIndex(points, near_radius=1.0, workers=2).query(queries)

    np.testing.assert_array_equal(actual, expected)


def test_far_lid_vertices_much_faster_than_exact_ckdtree():
    points = _perforated_bowl(2_000_000, seed=5)
    rng = np.random.default_rng(6)
    rho = rng.uniform(5, 40, 1500)
    phi = rng.random(1500) * 2 * np.pi
    lid = np.c_[rho * np.cos(phi), rho * np.sin(phi), rng.uniform(0, 5, 1500)]

    tree = cKDTree(points, leafsize=256)
    t0 = time.perf_counter()
    expected = tree.query(lid, k=1, workers=2)[1]
    t_ref = time.perf_counter() - t0

    index = NearestPointIndex(points, near_radius=1.0, workers=2)
    index.query(lid[:5])  # builds both trees outside the timed section (built once per object)
    t0 = time.perf_counter()
    actual = index.query(lid)
    t_new = time.perf_counter() - t0
    print(f"lid queries: cKDTree exact {t_ref:.3f} s, NearestPointIndex {t_new:.3f} s")

    np.testing.assert_array_equal(actual, expected)
    assert t_new * 5 < t_ref
