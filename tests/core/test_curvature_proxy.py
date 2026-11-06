"""Curvature proxy mesh: edge refinement (shape kept, already fine meshes untouched), curvature measured tile by tile."""

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from eval3d.core.proxy import curvature_proxy, refine_edges
from eval3d.pipeline.preprocess_gt import compute_gt_curvature


def _slab(x0, x1, y0, y1, z0, z1):
    return trimesh.creation.box(bounds=[[x0, y0, z0], [x1, y1, z1]])


def test_refine_bounds_edges_and_keeps_the_shape():
    m = _slab(0, 3, 0, 2, 0, 1)
    p = refine_edges(m, 0.25)
    assert p.edges_unique_length.max() <= 0.25
    assert p.is_watertight  # conforming: no T-junction
    assert p.area == pytest.approx(m.area) and p.volume == pytest.approx(m.volume)
    assert np.allclose(p.bounds, m.bounds)


def test_refine_leaves_a_fine_mesh_untouched():
    m = trimesh.creation.icosphere(subdivisions=4, radius=1.0)  # edges ~0.08
    assert refine_edges(m, 0.25) is m
    assert curvature_proxy(m, 0.25) is m and curvature_proxy(m, None) is m


def test_gt_curvature_on_a_coarse_sphere_uses_the_refined_proxy():
    m = trimesh.creation.icosphere(subdivisions=5, radius=10.0)  # edges ~0.37 > 0.15
    pts = m.vertices[::5]
    k = compute_gt_curvature(m, pts, radius=1.0, tube=0.5, max_edge=0.15)
    assert len(k) == len(pts) and k.dtype == np.float32 and np.isfinite(k).all()
    assert np.median(k) == pytest.approx(-0.1, rel=0.1)  # 0.197 on the coarse sphere itself
    assert len(curvature_proxy(m, 0.15).vertices) > len(m.vertices)


def test_proxy_curvature_is_computed_once_and_cached(tmp_path):
    from eval3d.config import Config
    from eval3d.pipeline.preprocess_gt import proxy_curvature
    gt_src = tmp_path / "src" / "obj"
    gt_src.mkdir(parents=True)
    trimesh.creation.icosphere(subdivisions=5, radius=10.0).export(gt_src / "gt.ply")  # edges ~0.37
    config = Config(dataset={"name": "t", "objects": ["obj"]},
                    paths={"data_root": str(tmp_path / "data" / "{object}"), "eval_root": str(tmp_path / "eval" / "{object}"),
                           "gt_root": str(tmp_path / "src" / "{object}")},
                    evaluation={"curvature_max_edge": 0.15})
    pts, k = proxy_curvature(config, "obj")
    f = config.get_gt_dir("obj") / "attributes" / "proxy_curvature_r1_R0.5_e0.15.npz"
    assert f.exists()
    assert len(k) == len(pts) and len(pts) == len(curvature_proxy(trimesh.load(gt_src / "gt.ply"), 0.15).vertices)
    assert np.median(k) == pytest.approx(-0.1, rel=0.1)
    stamp = f.stat().st_mtime_ns
    pts2, k2 = proxy_curvature(config, "obj")  # cached: nothing rewritten, same values
    assert f.stat().st_mtime_ns == stamp
    assert np.array_equal(pts2, pts) and np.array_equal(k2, k)


def test_tiled_curvature_matches_the_whole_mesh_run():
    from eval3d.core.proxy import tiled_curvature
    from eval3d.core.vcm import compute_vertex_curvature
    m = trimesh.creation.icosphere(subdivisions=4, radius=10.0)  # edges ~0.74: refined in each tile
    whole = curvature_proxy(m, 0.3)
    k_whole = compute_vertex_curvature(whole)
    pts, k = tiled_curvature(m, max_edge=0.3, tile=8.0)
    d, i = cKDTree(pts).query(whole.vertices)
    assert len(pts) == len(whole.vertices) and d.max() < 1e-9  # same proxy vertices, each kept once
    assert np.median(np.abs(k[i] - k_whole)) < 0.01 and np.median(k) == pytest.approx(np.median(k_whole), abs=0.005)
