"""The fast GT-attribute paths (vectorised curvature, in-view-only chunked visibility)
must give the same values as the former per-vertex / all-points implementations."""

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from src.core.mesh import compute_vertex_curvature
from src.core.visibility import compute_visibility_count, ray_visibility_check, check_points_in_mask
from src.core.camera import get_camera_centers


def _reference_curvature(mesh, radius):
    """Verbatim copy of the former per-vertex loop (src/core/mesh.py before the speed-up)."""
    tree = cKDTree(mesh.vertices)
    curvature = np.zeros(len(mesh.vertices), dtype=np.float32)
    normals = mesh.vertex_normals
    for i, idx in enumerate(tree.query_ball_point(mesh.vertices, r=radius)):
        if len(idx) < 6:
            continue
        pts_centered = mesh.vertices[idx] - mesh.vertices[i]
        normal = normals[i]
        tangent1 = np.cross(normal, [1, 0, 0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0, 1, 0])
        tangent1 /= np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)
        u, v, w = pts_centered @ tangent1, pts_centered @ tangent2, pts_centered @ normal
        params, _, _, _ = np.linalg.lstsq(np.column_stack([u**2, u*v, v**2]), w, rcond=None)
        a, b, c = params
        tr, det = 2*a + 2*c, 4*a*c - b**2
        disc = max(0, tr**2 - 4*det)
        k1, k2 = (tr + np.sqrt(disc)) / 2, (tr - np.sqrt(disc)) / 2
        curvature[i] = k1 if abs(k1) > abs(k2) else k2
    return curvature


def _bumpy_mesh():
    m = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    rng = np.random.default_rng(0)
    v = m.vertices * (1 + 0.03 * np.sin(3 * m.vertices[:, :1]) + 0.002 * rng.standard_normal((len(m.vertices), 1)))
    return trimesh.Trimesh(v, m.faces, process=False)


def test_vectorised_curvature_matches_reference_loop():
    m = _bumpy_mesh()
    ref = _reference_curvature(m, radius=2.0)
    fast = compute_vertex_curvature(m, radius=2.0, batch_size=257)       # odd batch: boundaries
    assert np.allclose(fast, ref, rtol=1e-4, atol=1e-6)


def test_curvature_on_vertex_subset_only():
    m = _bumpy_mesh()
    full = compute_vertex_curvature(m, radius=2.0)
    sub = np.array([5, 17, 17, 400, len(m.vertices) - 1])
    part = compute_vertex_curvature(m, radius=2.0, vertex_indices=sub)
    assert np.allclose(part[sub], full[sub], rtol=1e-6, atol=1e-9)
    others = np.setdiff1d(np.arange(len(m.vertices)), sub)
    assert not part[others].any()


def _scene(tmp_path):
    """Sphere + an occluding plate in front of camera 0; three cameras around."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    plate = trimesh.creation.box(extents=(0.8, 0.8, 0.05)); plate.apply_translation((0.3, 0.2, 2.0))
    mesh = trimesh.util.concatenate([sphere, plate])
    K = np.array([[300, 0, 160, 0], [0, 300, 120, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    Ps, Rts, names = [], [], []
    for k, ang in enumerate((0.0, 2.0, 4.0)):
        C = np.array([5 * np.sin(ang), 0.3, 5 * np.cos(ang)])
        z = -C / np.linalg.norm(C); x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x); y = np.cross(z, x)
        R = np.stack([x, y, z]); Rt = np.eye(4); Rt[:3, :3] = R; Rt[:3, 3] = -R @ C
        Ps.append(K @ Rt); Rts.append(Rt)
        mask = np.zeros((240, 320), np.uint8); mask[40:200, 60:260] = 255            # partial mask
        Image.fromarray(mask).save(tmp_path / f"{k:03d}.png"); names.append(f"{k:03d}.png")
    pts = trimesh.sample.sample_surface(sphere, 4000, seed=1)[0]
    return pts, mesh, {"P": np.array(Ps), "Rt": np.array(Rts), "mask_names": names}


def test_visibility_count_matches_all_points_reference(tmp_path):
    pts, mesh, cams = _scene(tmp_path)
    centres = get_camera_centers(cams["Rt"])
    ref = np.zeros(len(pts), np.int32)
    for k in range(3):
        mask = np.array(Image.open(tmp_path / cams["mask_names"][k])) > 127
        _, in_view = check_points_in_mask(pts, cams["P"][k], mask, 0)
        ref += (in_view & ray_visibility_check(pts, mesh, centres[k])).astype(np.int32)
    fast = compute_visibility_count(pts, mesh, cams, tmp_path, dilation_radius=0, show_progress=False,
                                    ray_chunk=333)
    assert np.array_equal(fast, ref)
    assert 0 < ref.sum() < 3 * len(pts)          # the scene does exercise occlusion and the masks


def test_parallel_curvature_identical_to_serial():
    m = _bumpy_mesh()
    serial = compute_vertex_curvature(m, radius=2.0, batch_size=300)
    parallel = compute_vertex_curvature(m, radius=2.0, batch_size=300, n_jobs=3)
    assert np.array_equal(serial, parallel)
    sub = np.arange(0, len(m.vertices), 7)
    assert np.array_equal(compute_vertex_curvature(m, radius=2.0, vertex_indices=sub, batch_size=100, n_jobs=2), 
                          compute_vertex_curvature(m, radius=2.0, vertex_indices=sub, batch_size=100))
