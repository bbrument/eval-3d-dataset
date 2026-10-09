"""Mesh utilities using trimesh."""

from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a mesh from a file.

    Args:
        path: Path to the mesh file (PLY, OBJ, etc.).

    Returns:
        Loaded trimesh object.
    """
    mesh = trimesh.load(str(path), force="mesh")
    # An empty or multi-geometry file makes trimesh return a Scene / list / PointCloud
    # (e.g. a failed COLMAP run writes a PLY with `element vertex 0`). Normalise to a
    # (possibly empty) Trimesh so callers can rely on .vertices/.faces and guard on
    # len()==0 rather than crash on a bare list having no `.update_faces`.
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
    elif isinstance(mesh, (list, tuple)):
        geoms = [g for g in mesh if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh()
    if len(mesh.faces) > 0:
        mesh.update_faces(mesh.nondegenerate_faces())
    return mesh


def save_mesh(mesh: trimesh.Trimesh, path: str | Path) -> None:
    """Save a mesh to a file.

    Args:
        mesh: Trimesh object to save.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path))


def get_vertices_faces(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Get vertices and faces from a mesh.

    Args:
        mesh: Trimesh object.

    Returns:
        Tuple of (vertices, faces) as numpy arrays.
    """
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def filter_mesh_by_vertex_mask(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> trimesh.Trimesh:
    """Filter a mesh keeping only vertices where mask is True.

    Uses cumsum-based reindexing to update face indices after vertex removal.

    Args:
        mesh: Input mesh.
        vertex_mask: Boolean array of shape (N,) where N is number of vertices.
            True = keep vertex, False = remove vertex.

    Returns:
        New mesh with filtered vertices and updated face indices.
    """
    vertex_mask = np.asarray(vertex_mask, dtype=bool)

    # Keep faces where all vertices are in the mask
    valid_faces_mask = vertex_mask[mesh.faces].all(axis=1)
    valid_faces = mesh.faces[valid_faces_mask]

    # Compute index shift using cumsum
    # For each vertex, shift[i] = number of removed vertices before index i
    shift = np.cumsum(~vertex_mask)

    # Adjust face indices: subtract the shift for each vertex index
    adjusted_faces = valid_faces - shift[valid_faces]

    # Create new mesh with filtered vertices
    new_vertices = mesh.vertices[vertex_mask]
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=adjusted_faces, process=False)

    # Clean up degenerate faces
    new_mesh.update_faces(new_mesh.nondegenerate_faces())

    return new_mesh


# Shared read-only state of the curvature workers (inherited through fork, never pickled).
_CURVATURE_SHARED: dict = {}


def _curvature_batch(ids: np.ndarray, radius: float, workers: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Quadric-fit curvature of one batch of vertex ids -> (ids with >= 6 neighbours, values)."""
    from itertools import chain

    V, normals, tree = _CURVATURE_SHARED["V"], _CURVATURE_SHARED["normals"], _CURVATURE_SHARED["tree"]
    neighbors = tree.query_ball_point(V[ids], r=radius, workers=workers)
    counts = np.fromiter((len(x) for x in neighbors), dtype=np.int64, count=len(neighbors))
    keep = counts >= 6
    if not keep.any():
        return ids[:0], np.zeros(0)
    ids, counts = ids[keep], counts[keep]
    flat = np.fromiter(chain.from_iterable(neighbors[i] for i in np.flatnonzero(keep)),
                       dtype=np.int64, count=int(counts.sum()))
    owner = np.repeat(np.arange(len(ids)), counts)

    normal = normals[ids]
    tangent1 = np.where((np.abs(normal[:, 0]) < 0.9)[:, None],
                        np.cross(normal, [1.0, 0.0, 0.0]), np.cross(normal, [0.0, 1.0, 0.0]))
    tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True)
    tangent2 = np.cross(normal, tangent1)

    pts = V[flat] - V[ids][owner]
    u = np.einsum("ij,ij->i", pts, tangent1[owner])
    v = np.einsum("ij,ij->i", pts, tangent2[owner])
    w = np.einsum("ij,ij->i", pts, normal[owner])
    f = (u * u, u * v, v * v)

    seg = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ata = np.empty((len(ids), 3, 3))
    for r in range(3):
        for c in range(r, 3):
            ata[:, r, c] = ata[:, c, r] = np.add.reduceat(f[r] * f[c], seg)
    atw = np.stack([np.add.reduceat(f[r] * w, seg) for r in range(3)], axis=1)
    a, b, c = np.einsum("bij,bj->bi", np.linalg.pinv(ata), atw).T

    tr = 2 * a + 2 * c
    det = 4 * a * c - b ** 2
    disc = np.sqrt(np.maximum(0, tr ** 2 - 4 * det))
    k1 = (tr + disc) / 2
    k2 = (tr - disc) / 2
    return ids, np.where(np.abs(k1) > np.abs(k2), k1, k2)


def _curvature_batch_star(args):
    return _curvature_batch(*args)


def compute_vertex_curvature(
    mesh: trimesh.Trimesh,
    radius: float | None = None,
    vertex_indices: np.ndarray | None = None,
    batch_size: int = 20_000,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute maximum principal curvature at each vertex.

    Uses a neighborhood-based estimation to highlight concave/convex regions:
    for each vertex, the quadric w = a u^2 + b uv + c v^2 is least-squares fitted
    (in its tangent frame) to ALL vertices within ``radius``; vertices with fewer
    than 6 neighbours get 0. If radius is None, it defaults to 5x the average
    edge length.

    Vectorised: each batch of vertices is solved at once through the 3x3 normal
    equations and a pseudo-inverse (= the minimum-norm least-squares solution of
    the former per-vertex ``np.linalg.lstsq`` loop, same values). The cost is
    ~N_vertices x neighbours (fixed radius): ~1e11 vertex-neighbour pairs for the
    densest scans (12_assiette: 0.05 mm edges, 7 400 neighbours/vertex, ~11 h on
    one core), hence ``n_jobs`` worker processes over the batches (same values).

    Args:
        mesh: Input mesh.
        radius: Spatial radius for neighborhood search.
        vertex_indices: Only compute these vertices (others are 0). Default: all.
        batch_size: Vertices per vectorised batch (memory ~ batch x neighbours
            x ~150 B per worker).
        n_jobs: Worker processes (fork). 1 = in-process.

    Returns:
        Array of signed maximum principal curvature values, shape (N,).
    """
    from scipy.spatial import cKDTree

    n_vertices = len(mesh.vertices)
    if n_vertices == 0:
        return np.array([], dtype=np.float32)

    # Estimate default radius if not provided (5x avg edge length)
    if radius is None:
        edges = mesh.edges_unique
        edge_lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
        avg_edge = np.mean(edge_lengths)
        radius = avg_edge * 5.0
        print(f"  Estimating curvature with radius={radius:.3f} mm (5x avg edge)")

    V = np.asarray(mesh.vertices, dtype=np.float64)
    _CURVATURE_SHARED.update(V=V, normals=np.asarray(mesh.vertex_normals, dtype=np.float64), tree=cKDTree(V))
    todo = np.arange(n_vertices) if vertex_indices is None else np.unique(np.asarray(vertex_indices))
    batches = [todo[s:s + batch_size] for s in range(0, len(todo), batch_size)]
    curvature = np.zeros(n_vertices, dtype=np.float32)

    def report(i):
        if len(batches) > 1 and (i % 50 == 0 or i == len(batches) - 1):
            print(f"  Curvature batch {i+1}/{len(batches)}", flush=True)

    try:
        if n_jobs <= 1:
            for i, ids in enumerate(batches):
                report(i)
                done, values = _curvature_batch(ids, radius)
                curvature[done] = values
        else:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(n_jobs) as pool:
                jobs = ((ids, radius, 1) for ids in batches)
                for i, (done, values) in enumerate(pool.imap_unordered(_curvature_batch_star, jobs)):
                    report(i)
                    curvature[done] = values
    finally:
        _CURVATURE_SHARED.clear()

    return curvature
