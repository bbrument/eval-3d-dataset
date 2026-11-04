"""Proxy mesh on which the GT curvature is computed (the GT itself is left untouched).

Edges longer than `max_edge` split at their midpoint until none is left. The VCM inflates |k| next to large
triangles (CT GT, up to 10 mm edges) and is biased upwards on coarse meshes (sphere of radius 10, r = 1: |k| 0.128
at edge 0.19, 0.108 at 0.09 for 0.1). 0.15 = the density of the non-CT GT (edge median 0.105, p99 0.155), so all GT
are measured alike. The shape is unchanged; already fine meshes are returned as is. The proxy of a large GT does not
fit in memory (spine 12 M vertices, 13 GB; plant 63 M): tiled_curvature builds and measures it tile by tile.
"""
import numpy as np
import trimesh


def refine_edges(mesh, max_edge: float):
    """Split every edge longer than max_edge at its midpoint (conforming: both triangles of an edge split it)."""
    V, F = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
    if len(F) == 0 or mesh.edges_unique_length.max() <= max_edge:
        return mesh
    idx = np.int32 if len(V) < 2**30 else np.int64  # halves the face arrays (refined meshes stay < 2^30 vertices)
    F = F.astype(idx)
    while True:
        E = np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)  # edge j of a face: (v_j, v_j+1)
        key, inv = np.unique(E[:, 0].astype(np.int64) * len(V) + E[:, 1], return_inverse=True)
        del E
        long = np.linalg.norm(V[key // len(V)] - V[key % len(V)], axis=1) > max_edge
        if not long.any():
            return trimesh.Trimesh(V, F, process=False)
        mid = np.full(len(key), -1, idx)
        mid[long] = len(V) + np.arange(long.sum())
        a, b = key[long] // len(V), key[long] % len(V)
        V = np.vstack([V, (V[a] + V[b]) / 2])
        M = mid[inv.reshape(-1, 3)]
        del key, inv, mid, long, a, b
        F = _split_faces(F, M)


def _split_faces(F, M):
    """Triangulate each face (a, b, c) given the midpoints M of its edges ab, bc, ca (-1 = edge not split)."""
    s = M >= 0
    k = s.sum(1)
    out = [F[k == 0]]
    for n in (1, 2, 3):
        f, m, ss = F[k == n], M[k == n], s[k == n]
        # rotate so that the split edge is ab (n = 1) or the unsplit one is ca (n = 2)
        rot = np.argmax(ss, 1) if n == 1 else (np.argmin(ss, 1) + 1) % 3 if n == 2 else np.zeros(len(f), int)
        idx = (rot[:, None] + np.arange(3)) % 3
        f, m = np.take_along_axis(f, idx, 1), np.take_along_axis(m, idx, 1)
        a, b, c, m0, m1, m2 = f[:, 0], f[:, 1], f[:, 2], m[:, 0], m[:, 1], m[:, 2]
        tris = {1: [(a, m0, c), (m0, b, c)],
                2: [(m0, b, m1), (a, m0, m1), (a, m1, c)],
                3: [(a, m0, m2), (m0, b, m1), (m2, m1, c), (m0, m1, m2)]}[n]
        out += [np.column_stack(t) for t in tris]
    return np.concatenate(out)


def curvature_proxy(mesh, max_edge: float | None = 0.15, verbose: bool = False):
    """GT mesh -> proxy mesh for the curvature: edges refined to max_edge (None = GT mesh as is).
    Connected components share no edge: refined one by one (memory), then put back together."""
    if max_edge is None or len(mesh.faces) == 0 or mesh.edges_unique_length.max() <= max_edge:
        proxy = mesh
    else:
        proxy = trimesh.util.concatenate([refine_edges(c, max_edge) for c in mesh.split(only_watertight=False)])
    if verbose:
        print(f"  Curvature proxy: {len(mesh.vertices):,} -> {len(proxy.vertices):,} vertices (max edge {max_edge})",
              flush=True)
    return proxy


def tiled_curvature(mesh, max_edge: float | None = 0.15, radius: float = 1.0, tube: float = 0.5,
                    tile: float = 30.0, verbose: bool = False):
    """VCM curvature on the vertices of curvature_proxy(mesh), tile by tile: the memory is set by the tile, not the mesh.

    Each cube of side `tile` gets the GT faces reaching it or its margin, refined alone (the midpoint split of an edge
    only depends on that edge: same proxy vertices as the whole mesh), and keeps the vertices of its cube. The margin
    holds all that the curvature of a vertex reads: connected ball r, Voronoi cells of the tube up to 2R, one edge.
    shortcut: the tube sampling is drawn per tile (its step follows the tile's vertex density), so the result matches
    the whole-mesh run statistically, not bit for bit; seed the samples per coarse cell if exactness is needed.

    Returns:
        (points (n, 3), curvature (n,) float32): every proxy vertex once, as compute_vertex_curvature signs it.
    """
    from .vcm import compute_vertex_curvature
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    edge = max_edge if max_edge is not None else mesh.edges_unique_length.max()
    margin = radius + 2 * tube + 2 * edge
    lo = V.min(0)
    T = V[F]
    i0 = np.floor((T.min(1) - margin - lo) / tile).astype(np.int64).clip(0)
    i1 = np.floor((T.max(1) + margin - lo) / tile).astype(np.int64)
    del T
    dims = i1.max(0) + 1
    span = (i1 - i0).max() + 1
    face, key = [], []
    for d in np.ndindex(span, span, span):  # (face, tile) pairs: every tile met by the face's box + margin
        i = i0 + d
        ok = (i <= i1).all(1)
        face.append(np.flatnonzero(ok))
        key.append(np.ravel_multi_index(i[ok].T, dims))
    face, key = np.concatenate(face), np.concatenate(key)
    order = np.argsort(key, kind="stable")
    face, key = face[order], key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    points, curv = [], []
    for n, (s, e) in enumerate(zip(starts, np.r_[starts[1:], len(key)])):
        t_lo = lo + np.array(np.unravel_index(key[s], dims)) * tile
        sub = trimesh.Trimesh(V, F[face[s:e]], process=False)
        sub.remove_unreferenced_vertices()
        p = curvature_proxy(sub, max_edge)
        inner = ((p.vertices >= t_lo - margin) & (p.vertices < t_lo + tile + margin)).all(1)
        p.update_faces(inner[p.faces].any(1))  # large GT triangles reach far outside the margin
        p.remove_unreferenced_vertices()
        core = ((p.vertices >= t_lo) & (p.vertices < t_lo + tile)).all(1)
        if not core.any():
            continue
        k = compute_vertex_curvature(p, radius, tube)
        points.append(p.vertices[core])
        curv.append(k[core])
        if verbose:
            print(f"  tile {n + 1}/{len(starts)}: {len(p.vertices):,} proxy vertices, {core.sum():,} kept", flush=True)
    return np.concatenate(points), np.concatenate(curv)
