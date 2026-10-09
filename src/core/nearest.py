"""Exact nearest-neighbour index over a dense GT point cloud.

scipy's cKDTree bounds its cells by split planes, not by the points they hold. A
query far from a dense concave surface (e.g. reconstruction vertices inside the
hemispherical colander, tens of mm from every GT point) intersects most cells and
degenerates into a near brute-force scan: up to ~15 ms per vertex over the 23 M GT
points of 10_colander, i.e. 4-6 h per visualization.

``NearestPointIndex.query`` returns exactly ``cKDTree(points, leafsize).query(v, k=1)[1]``
(the historical code), much faster:

* vertices within ``near_radius`` of the points: bounded query of that same cKDTree
  (cheap, the bound only prunes cells that cannot hold the nearest point);
* the other, far vertices: scikit-learn's KDTree, whose nodes keep tight bounding
  boxes, so the search stays local;
* ties: both paths ask for the 2 nearest points; when they are equidistant (up to
  ``TIE_RTOL``), the plain cKDTree query is re-run for that vertex so that the
  equidistant point is chosen exactly as before.

The result does not depend on ``near_radius``; only the speed does.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

#: Relative gap under which the two nearest points count as equidistant.
TIE_RTOL = 1e-12
#: Chunks per worker for the threaded far query (load balancing).
_CHUNKS_PER_WORKER = 16


def default_workers() -> int:
    """CPUs reserved by Slurm for this task, 1 outside Slurm."""
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))


class NearestPointIndex:
    """Exact nearest-point lookup, fast also for queries far from the points.

    Args:
        points: (N, 3) reference points (e.g. gt_pcd).
        near_radius: Vertices closer than this (unit of ``points``) use the cKDTree,
            the others the tight-box KD-tree. Affects speed only.
        workers: Threads per query (default: SLURM_CPUS_PER_TASK, else 1).
        leafsize: cKDTree leaf size; it fixes how ties are broken (16 = scipy
            default = historical renders).
        far_leaf_size: scikit-learn KDTree leaf size.
    """

    def __init__(
        self,
        points: np.ndarray,
        near_radius: float = 1.0,
        workers: int | None = None,
        leafsize: int = 16,
        far_leaf_size: int = 40,
    ):
        self._points = np.ascontiguousarray(points, dtype=np.float64)
        self._near_radius = float(near_radius)
        self._workers = workers if workers is not None else default_workers()
        self._tree = cKDTree(self._points, leafsize=leafsize)
        self._far_leaf_size = far_leaf_size
        self._far_tree = None
        self.last_stats: dict[str, int] = {}

    def query(self, vertices: np.ndarray) -> np.ndarray:
        """Return, for each vertex, the index of its nearest point."""
        vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        if len(vertices) == 0 or len(self._points) < 2:
            return self._exact(vertices)
        dist, idx = self._tree.query(
            vertices, k=2, distance_upper_bound=self._near_radius, workers=self._workers
        )
        far = ~np.isfinite(dist[:, 0])
        if far.any():
            dist[far], idx[far] = self._query_far(vertices[far])
        indices = idx[:, 0].copy()
        tie = dist[:, 1] <= dist[:, 0] * (1.0 + TIE_RTOL)
        if tie.any():
            indices[tie] = self._exact(vertices[tie])
        self.last_stats = {
            "vertices": int(len(vertices)),
            "far": int(far.sum()),
            "ties": int(tie.sum()),
        }
        return indices

    def _exact(self, vertices: np.ndarray) -> np.ndarray:
        """Plain cKDTree query, i.e. the historical code path."""
        if len(vertices) == 0:
            return np.empty(0, dtype=np.intp)
        return self._tree.query(vertices, k=1, workers=self._workers)[1]

    def _query_far(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._far_tree is None:
            from sklearn.neighbors import KDTree

            self._far_tree = KDTree(self._points, leaf_size=self._far_leaf_size)
        n_chunks = min(len(vertices), self._workers * _CHUNKS_PER_WORKER)
        chunks = np.array_split(np.arange(len(vertices)), n_chunks)

        def run(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return self._far_tree.query(vertices[chunk], k=2, return_distance=True)

        # scikit-learn releases the GIL in the per-point search loop.
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            parts = list(pool.map(run, chunks))
        dist = np.concatenate([p[0] for p in parts])
        idx = np.concatenate([p[1] for p in parts]).astype(np.intp, copy=False)
        return dist, idx
