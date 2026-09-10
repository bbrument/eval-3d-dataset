"""Tests for core/alignment.py and the optional data-to-GT alignment step in evaluate()."""

import json
import math

import numpy as np
import pytest
import trimesh

from src.config import Config, DatasetConfig, EvaluationConfig, PathsConfig
from src.core.alignment import align_sim3, apply_transform, umeyama
from src.core.mesh import load_mesh, save_mesh
from src.core.metrics import compute_distances, compute_metrics
from src.core.sampling import downsample_pcd, upsample_mesh
from src.pipeline.evaluate import evaluate


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation about `axis` by `angle_deg`."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = math.radians(angle_deg)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def _sim3(scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T


def _rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def _decompose(T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (scale, R, t) from a Sim(3) matrix."""
    A = T[:3, :3]
    s = float(np.cbrt(np.linalg.det(A)))
    return s, A / s, T[:3, 3]


# A box with unequal sides has no rotational symmetry, so a Sim(3) is identifiable.
_BOX_EXTENTS = (2.0, 1.0, 0.5)

# Known perturbation used throughout: scale x1.3, 15 degrees, translation.
_PERTURB_SCALE = 1.3
_PERTURB_R = _rotation_matrix([0.3, 1.0, 0.2], 15.0)
_PERTURB_T = np.array([0.4, -0.25, 0.15])
_PERTURB = _sim3(_PERTURB_SCALE, _PERTURB_R, _PERTURB_T)


def _box_surface_points(n: int, seed: int) -> np.ndarray:
    mesh = trimesh.creation.box(extents=_BOX_EXTENTS)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float64)


# ---------------------------------------------------------------------------
# (a) Umeyama
# ---------------------------------------------------------------------------

class TestUmeyama:
    """Closed-form Sim(3) / SE(3) on exact correspondences."""

    def test_recovers_known_sim3(self):
        src = _box_surface_points(500, seed=0)
        dst = apply_transform(src, _PERTURB)

        T = umeyama(src, dst, with_scale=True)

        np.testing.assert_allclose(T, _PERTURB, atol=1e-6)
        np.testing.assert_allclose(apply_transform(src, T), dst, atol=1e-6)

    def test_recovers_known_se3_without_scale(self):
        src = _box_surface_points(500, seed=1)
        T_true = _sim3(1.0, _PERTURB_R, _PERTURB_T)
        dst = apply_transform(src, T_true)

        T = umeyama(src, dst, with_scale=False)

        np.testing.assert_allclose(T, T_true, atol=1e-6)
        s, _, _ = _decompose(T)
        assert s == pytest.approx(1.0, abs=1e-9)

    def test_handles_reflection_degeneracy(self):
        """Nearly-planar sets must still yield a proper rotation (det=+1)."""
        rng = np.random.default_rng(3)
        src = rng.uniform(-1, 1, size=(200, 3))
        src[:, 2] *= 1e-3  # almost planar
        dst = apply_transform(src, _PERTURB)

        T = umeyama(src, dst, with_scale=True)
        _, R, _ = _decompose(T)

        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(apply_transform(src, T), dst, atol=1e-5)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            umeyama(np.zeros((5, 3)), np.zeros((4, 3)), with_scale=True)

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError):
            umeyama(np.zeros((2, 3)), np.zeros((2, 3)), with_scale=True)


# ---------------------------------------------------------------------------
# apply_transform
# ---------------------------------------------------------------------------

class TestApplyTransform:

    def test_points_are_not_mutated(self):
        pts = _box_surface_points(50, seed=4)
        before = pts.copy()
        out = apply_transform(pts, _PERTURB)
        np.testing.assert_array_equal(pts, before)
        assert out.shape == pts.shape

    def test_mesh_returns_new_mesh_with_transformed_vertices(self):
        mesh = trimesh.creation.box(extents=_BOX_EXTENTS)
        before = mesh.vertices.copy()
        out = apply_transform(mesh, _PERTURB)

        assert out is not mesh
        np.testing.assert_array_equal(mesh.vertices, before)
        np.testing.assert_array_equal(out.faces, mesh.faces)
        np.testing.assert_allclose(out.vertices, apply_transform(before, _PERTURB), atol=1e-9)

    def test_identity_is_noop(self):
        pts = _box_surface_points(50, seed=5)
        np.testing.assert_allclose(apply_transform(pts, np.eye(4)), pts)


# ---------------------------------------------------------------------------
# (b) Robust ICP
# ---------------------------------------------------------------------------

def _make_icp_problem(seed: int, outlier_fraction: float = 0.2):
    """dst = GT box surface; src = a subset of it in a perturbed frame, plus outliers.

    Returns (src, dst, T_true) where T_true maps src -> dst.
    """
    rng = np.random.default_rng(seed)
    dst = _box_surface_points(4000, seed=seed)

    inliers_dst = dst[rng.choice(len(dst), size=2000, replace=False)]
    inliers_src = apply_transform(inliers_dst, np.linalg.inv(_PERTURB))

    n_out = int(round(outlier_fraction * len(inliers_src) / (1.0 - outlier_fraction)))
    lo, hi = inliers_src.min(axis=0), inliers_src.max(axis=0)
    span = hi - lo
    outliers = rng.uniform(lo - span, hi + span, size=(n_out, 3))

    src = np.concatenate([inliers_src, outliers], axis=0)
    src = src[rng.permutation(len(src))]
    return src, dst, _PERTURB


class TestAlignSim3:

    def test_recovers_perturbation_with_outliers(self):
        src, dst, T_true = _make_icp_problem(seed=10)

        T, stats = align_sim3(src, dst, scale=True, robust=True)

        s_est, R_est, t_est = _decompose(T)
        s_true, R_true, t_true = _decompose(T_true)

        assert s_est == pytest.approx(s_true, rel=1e-2)
        assert _rotation_error_deg(R_est, R_true) < 0.5
        assert np.linalg.norm(t_est - t_true) < 0.02 * max(_BOX_EXTENTS)

        # Stats are informative and consistent with a converged solve.
        assert stats["residual_median_after"] < stats["residual_median_before"]
        assert stats["residual_median_after"] < 1e-2
        assert 0.7 <= stats["inlier_fraction"] <= 0.9  # ~80% inliers by construction
        assert stats["scale"] == pytest.approx(s_est, rel=1e-9)
        assert 1 <= stats["n_iterations"] <= stats["max_iterations"]

    def test_se3_mode_keeps_unit_scale(self):
        src, dst, _ = _make_icp_problem(seed=11)
        # Undo the scale so an SE(3) solution exists.
        src = src * _PERTURB_SCALE

        T, stats = align_sim3(src, dst, scale=False, robust=True)

        s_est, R_est, _ = _decompose(T)
        assert s_est == pytest.approx(1.0, abs=1e-9)
        assert stats["scale"] == pytest.approx(1.0, abs=1e-9)
        assert _rotation_error_deg(R_est, _PERTURB_R) < 0.5

    def test_init_is_used(self):
        """Starting from the exact answer should converge in very few iterations."""
        src, dst, T_true = _make_icp_problem(seed=12)

        T, stats = align_sim3(src, dst, init=T_true, scale=True, robust=True)

        s_est, R_est, _ = _decompose(T)
        assert s_est == pytest.approx(_PERTURB_SCALE, rel=1e-3)
        assert _rotation_error_deg(R_est, _PERTURB_R) < 0.1
        assert stats["n_iterations"] <= 5

    def test_iterations_are_bounded(self):
        src, dst, _ = _make_icp_problem(seed=13)
        T, stats = align_sim3(src, dst, max_iterations=3)
        assert T.shape == (4, 4)
        assert stats["n_iterations"] <= 3
        assert stats["max_iterations"] == 3

    def test_non_robust_is_pulled_by_outliers(self):
        """Without rejection, 20% far outliers must degrade the scale estimate."""
        src, dst, _ = _make_icp_problem(seed=14)
        T_robust, _ = align_sim3(src, dst, robust=True)
        T_plain, _ = align_sim3(src, dst, robust=False)

        s_robust, _, _ = _decompose(T_robust)
        s_plain, _, _ = _decompose(T_plain)
        assert abs(s_robust - _PERTURB_SCALE) < abs(s_plain - _PERTURB_SCALE)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            align_sim3(np.zeros((2, 3)), np.zeros((10, 3)))
        with pytest.raises(ValueError):
            align_sim3(np.zeros((10, 2)), np.zeros((10, 3)))
        with pytest.raises(ValueError):
            align_sim3(np.zeros((10, 3)), np.zeros((10, 3)), init=np.eye(3))


# ---------------------------------------------------------------------------
# (c)/(d) evaluate() integration
# ---------------------------------------------------------------------------

_DENSITY = 0.1
_MAX_DIST = 0.5
_THRESHOLDS = [0.05, 0.1, 0.2]


def _gt_box_pcd() -> np.ndarray:
    mesh = trimesh.creation.box(extents=_BOX_EXTENTS)
    pcd = upsample_mesh(mesh.vertices, mesh.faces, density=0.05, n_jobs=1)
    return downsample_pcd(pcd, density=0.05, seed=42).astype(np.float32)


def _recon_mesh() -> trimesh.Trimesh:
    """A GT-like box with non-similarity noise, so no Sim(3) can make it exact."""
    mesh = trimesh.creation.box(extents=_BOX_EXTENTS)
    for _ in range(3):
        mesh = mesh.subdivide()
    rng = np.random.default_rng(7)
    noisy = mesh.vertices + rng.normal(scale=0.01, size=mesh.vertices.shape)
    return trimesh.Trimesh(vertices=noisy, faces=mesh.faces, process=False)


def _build_eval_tree(tmp_path, mesh: trimesh.Trimesh, method: str = "m") -> Config:
    config = Config(
        dataset=DatasetConfig(name="t", objects=["obj"], methods=[method]),
        paths=PathsConfig(
            data_root=str(tmp_path / "data" / "{object}"),
            eval_root=str(tmp_path / "eval" / "{object}"),
        ),
        evaluation=EvaluationConfig(
            downsample_density=_DENSITY, max_dist=_MAX_DIST, fscore_thresholds=_THRESHOLDS
        ),
    )
    gt_dir = config.get_gt_dir("obj")
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.save(gt_dir / "gt_pcd.npy", _gt_box_pcd())
    save_mesh(mesh, config.get_cleaned_mesh_path("obj", method))
    return config


def _legacy_metrics(config: Config, seed: int = 42) -> dict:
    """The evaluate() computation as it stood before alignment existed."""
    gt_pcd = np.load(config.get_gt_dir("obj") / "gt_pcd.npy")
    mesh = load_mesh(config.get_cleaned_mesh_path("obj", "m"))
    density = config.evaluation.downsample_density
    data_pcd = upsample_mesh(mesh.vertices, mesh.faces, density)
    data_pcd = data_pcd[data_pcd[:, 2] > gt_pcd[:, 2].min()]
    data_pcd = data_pcd[~np.isnan(data_pcd).any(axis=1)]
    data_down = downsample_pcd(data_pcd, density, shuffle=True, seed=seed)
    dist_data2gt, _ = compute_distances(data_down, gt_pcd)
    dist_gt2data, _ = compute_distances(gt_pcd, data_down)
    return compute_metrics(
        dist_data2gt, dist_gt2data, config.evaluation.fscore_thresholds, config.evaluation.max_dist
    )


class TestEvaluateAlignmentDisabled:

    def test_default_config_has_alignment_disabled(self):
        config = Config(
            dataset=DatasetConfig(name="t"), paths=PathsConfig(data_root="/x/{object}")
        )
        assert config.alignment.enabled is False

    def test_metrics_identical_to_legacy_pipeline(self, tmp_path):
        """(c) alignment.enabled=False must reproduce the historical numbers exactly."""
        mesh = _recon_mesh()
        config = _build_eval_tree(tmp_path, mesh)
        expected = _legacy_metrics(config)

        metrics = evaluate(config, "obj", "m", force=True)

        for key, value in expected.items():
            assert key in metrics
            if isinstance(value, list):
                np.testing.assert_array_equal(np.array(metrics[key]), np.array(value))
            else:
                assert metrics[key] == value, key
        assert "alignment_residual_median" not in metrics
        assert not (config.get_eval_dir("obj", "m") / "alignment.json").exists()


class TestEvaluateAlignmentEnabled:

    def test_perturbed_mesh_recovers_unperturbed_chamfer(self, tmp_path):
        """(d) A Sim(3)-perturbed reconstruction scores ~like the unperturbed one."""
        mesh = _recon_mesh()
        perturbed = apply_transform(mesh, _PERTURB)

        config_ref = _build_eval_tree(tmp_path / "ref", mesh)
        reference = evaluate(config_ref, "obj", "m", force=True)

        config_raw = _build_eval_tree(tmp_path / "raw", perturbed)
        raw = evaluate(config_raw, "obj", "m", force=True)

        config_aligned = _build_eval_tree(tmp_path / "aligned", perturbed)
        config_aligned.alignment.enabled = True
        config_aligned.alignment.mode = "sim3"
        config_aligned.alignment.init = "centroid"
        aligned = evaluate(config_aligned, "obj", "m", force=True)

        # Sanity: unaligned, the perturbed mesh is far off (or unmeasurable).
        assert math.isnan(raw["chamfer"]) or raw["chamfer"] > 3 * reference["chamfer"]

        assert aligned["chamfer"] == pytest.approx(reference["chamfer"], rel=0.15)
        assert aligned["coverage"] == pytest.approx(reference["coverage"], abs=0.05)
        assert aligned["alignment_residual_median"] < 0.05
        assert aligned["alignment_residual_median"] == pytest.approx(reference["chamfer"], rel=0.5)

        # Sidecar with the transform and the solver statistics.
        alignment_path = config_aligned.get_eval_dir("obj", "m") / "alignment.json"
        assert alignment_path.exists()
        with open(alignment_path) as f:
            record = json.load(f)
        T = np.array(record["transform"])
        assert T.shape == (4, 4)
        s_est, R_est, _ = _decompose(T)
        assert s_est == pytest.approx(1.0 / _PERTURB_SCALE, rel=2e-2)
        assert _rotation_error_deg(R_est, _PERTURB_R.T) < 1.0
        for key in ("residual_median_before", "residual_median_after", "inlier_fraction",
                    "scale", "n_iterations"):
            assert key in record["stats"]
        assert record["mode"] == "sim3"
        assert record["init"] == "centroid"

    def test_se3_mode_cannot_fix_scale(self, tmp_path):
        """mode=se3 keeps scale at 1, so a scaled reconstruction stays wrong."""
        mesh = _recon_mesh()
        perturbed = apply_transform(mesh, _PERTURB)

        config = _build_eval_tree(tmp_path, perturbed)
        config.alignment.enabled = True
        config.alignment.mode = "se3"
        metrics = evaluate(config, "obj", "m", force=True)

        with open(config.get_eval_dir("obj", "m") / "alignment.json") as f:
            record = json.load(f)
        assert record["stats"]["scale"] == pytest.approx(1.0, abs=1e-9)
        assert math.isnan(metrics["chamfer"]) or metrics["chamfer"] > 0.05

    def test_camera_init_from_npy_centers(self, tmp_path):
        """init=cameras seeds the ICP with Umeyama on matching camera centres."""
        mesh = _recon_mesh()
        perturbed = apply_transform(mesh, _PERTURB)
        config = _build_eval_tree(tmp_path, perturbed)

        rng = np.random.default_rng(21)
        cams_dst = rng.uniform(-3, 3, size=(12, 3))
        # The reconstruction's cameras live in the mesh frame: GT cameras pushed
        # through the same perturbation as the mesh, plus a little pose noise.
        cams_src = apply_transform(cams_dst, _PERTURB) + rng.normal(
            scale=0.01, size=cams_dst.shape
        )
        src_path = tmp_path / "cams_src.npy"
        dst_path = tmp_path / "cams_dst.npy"
        np.save(src_path, cams_src)
        np.save(dst_path, cams_dst)

        config.alignment.enabled = True
        config.alignment.init = "cameras"
        config.alignment.cameras_src = str(src_path)
        config.alignment.cameras_dst = str(dst_path)
        metrics = evaluate(config, "obj", "m", force=True)

        with open(config.get_eval_dir("obj", "m") / "alignment.json") as f:
            record = json.load(f)
        assert record["init"] == "cameras"
        assert np.array(record["init_transform"]).shape == (4, 4)
        s_est, _, _ = _decompose(np.array(record["transform"]))
        assert s_est == pytest.approx(1.0 / _PERTURB_SCALE, rel=2e-2)
        assert metrics["alignment_residual_median"] < 0.05

    def test_camera_init_requires_paths(self, tmp_path):
        mesh = _recon_mesh()
        config = _build_eval_tree(tmp_path, mesh)
        config.alignment.enabled = True
        config.alignment.init = "cameras"
        with pytest.raises(ValueError):
            evaluate(config, "obj", "m", force=True)
