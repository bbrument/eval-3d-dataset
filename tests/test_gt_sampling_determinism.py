"""Reproducibility of the GT point-cloud sampling (gt_pcd.npy).

Regression tests for the non-deterministic gt_pcd bug: preprocess_gt used to call
``downsample_pcd`` without a seed, so the shuffle inside it fell back to
``np.random.default_rng(None)`` (seeded from OS entropy). Two runs on the same mesh
shuffled the upsampled cloud differently, the greedy radius selection then kept a
slightly different subset, and gt_pcd.npy drifted by ~0.04% between runs.

The fix threads ``config.evaluation.sampling_seed`` (default 42) into that call. These
tests pin the behaviour down with a tiny synthetic mesh (no cluster data required):

* same mesh + same seed  -> gt_pcd identical bit-for-bit,
* different seeds        -> different clouds,
* the defect itself      -> an *unseeded* downsample is non-deterministic (this is what
                            the pipeline did before the fix), and preprocess_gt now
                            actually forwards the configured seed to downsample_pcd
                            (this wiring test fails on the pre-fix code, where the seed
                            argument was None instead of the configured value).
"""

import inspect

import numpy as np
import pytest

from src.core.sampling import upsample_mesh, downsample_pcd
from src.config import Config, DatasetConfig, PathsConfig, EvaluationConfig


def _plane_grid(n: int = 6, spacing: float = 1.0):
    """A small triangulated plane grid: dense enough that downsampling drops points,
    so the shuffle order genuinely affects which points survive."""
    coords = np.arange(n) * spacing
    vertices = np.array(
        [[x, y, 0.0] for y in coords for x in coords], dtype=np.float64
    )
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b = j * n + i, j * n + i + 1
            c, d = (j + 1) * n + i, (j + 1) * n + i + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return vertices, np.array(faces)


def _sample_gt(seed, density: float = 0.3):
    """Reproduce the gt_pcd recipe from preprocess_gt: upsample then downsample."""
    vertices, faces = _plane_grid()
    pcd = upsample_mesh(vertices, faces, density=density, n_jobs=1)
    return downsample_pcd(pcd, density, seed=seed)


class TestGtPcdSamplingDeterminism:
    """The gt_pcd recipe (upsample + downsample) under an explicit seed."""

    def test_same_seed_is_bit_identical(self):
        """Same mesh + same seed -> identical gt_pcd on two runs, bit-for-bit."""
        first = _sample_gt(seed=42)
        second = _sample_gt(seed=42)

        # Bit-for-bit: identical shape, dtype and exact byte content.
        assert first.shape == second.shape
        assert first.dtype == second.dtype
        np.testing.assert_array_equal(first, second)
        assert first.tobytes() == second.tobytes()

    def test_different_seeds_give_different_clouds(self):
        """Different seeds -> different clouds (order of the greedy selection changes)."""
        a = _sample_gt(seed=42)
        b = _sample_gt(seed=7)

        # They may even differ in length; if not, the content must differ.
        assert not (a.shape == b.shape and np.array_equal(a, b))

    def test_defect_unseeded_downsample_is_nondeterministic(self):
        """Regression witness for the bug: without a seed, downsample_pcd draws from OS
        entropy, so two runs on the *same* cloud disagree. This is exactly what the
        pipeline did before a seed was threaded in, and why gt_pcd was not reproducible.
        """
        vertices, faces = _plane_grid()
        pcd = upsample_mesh(vertices, faces, density=0.3, n_jobs=1)

        unseeded_1 = downsample_pcd(pcd, 0.3)  # seed=None -> default_rng(None)
        unseeded_2 = downsample_pcd(pcd, 0.3)

        assert not (
            unseeded_1.shape == unseeded_2.shape
            and np.array_equal(unseeded_1, unseeded_2)
        ), "unseeded downsample was reproducible; the defect witness no longer holds"


class TestSamplingSeedConfig:
    """The new evaluation.sampling_seed configuration field."""

    def test_default_is_42(self):
        """Default seed is a fixed 42 so out-of-the-box runs are reproducible."""
        assert EvaluationConfig().sampling_seed == 42

    def test_matches_reconstruction_side_seed(self):
        """Consistency: the GT-sampling default matches the reconstruction-side default
        seed used in pipeline.evaluate (metrics['seed']=42), so GT and data clouds are
        drawn under the same seed."""
        from src.pipeline.evaluate import evaluate

        evaluate_seed_default = inspect.signature(evaluate).parameters["seed"].default
        assert evaluate_seed_default == 42
        assert EvaluationConfig().sampling_seed == evaluate_seed_default

    def test_configurable(self):
        """The seed can be overridden through config."""
        assert EvaluationConfig(sampling_seed=123).sampling_seed == 123


class TestPreprocessGtWiring:
    """preprocess_gt must actually forward the configured seed to downsample_pcd.

    This is the test that fails on the pre-fix code: there downsample_pcd was called
    with no seed, so the captured seed would be None rather than the configured value.
    """

    def _make_config(self, tmp_path, seed):
        eval_root = str(tmp_path / "{object}")
        return Config(
            dataset=DatasetConfig(name="synthetic", objects=["obj"]),
            paths=PathsConfig(data_root=str(tmp_path), eval_root=eval_root),
            evaluation=EvaluationConfig(sampling_seed=seed),
        )

    @pytest.mark.parametrize("seed", [42, 7])
    def test_preprocess_gt_forwards_config_seed(self, tmp_path, monkeypatch, seed):
        from src.pipeline import preprocess_gt as pg

        config = self._make_config(tmp_path, seed)

        # Put a tiny raw GT mesh where preprocess_gt looks for it.
        gt_dir = config.get_gt_dir("obj")
        gt_dir.mkdir(parents=True, exist_ok=True)
        vertices, faces = _plane_grid()
        import trimesh

        trimesh.Trimesh(vertices=vertices, faces=faces).export(gt_dir / "gt_mesh.ply")

        # Keep the test fast and focused: stub the heavy upsample, and capture the seed
        # that preprocess_gt hands to downsample_pcd.
        captured = {}

        def fake_upsample(vertices, faces, density, n_jobs=-1):
            return np.random.RandomState(0).rand(200, 3).astype(np.float32)

        def spy_downsample(pcd, density, shuffle=True, seed=None):
            captured["seed"] = seed
            return downsample_pcd(pcd, density, shuffle=shuffle, seed=seed)

        monkeypatch.setattr(pg, "upsample_mesh", fake_upsample)
        monkeypatch.setattr(pg, "downsample_pcd", spy_downsample)

        pg.preprocess_gt(
            config,
            "obj",
            compute_curvature=False,
            compute_visibility=False,
            clean_gt=False,
        )

        assert captured["seed"] == seed
