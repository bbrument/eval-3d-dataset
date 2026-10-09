"""Tests for pipeline/aggregate.py: JSON-safe output and method discovery.

Regression for the 2026-10-07 re-eval aggregation crash
(`TypeError: Object of type float32 is not JSON serializable`): the published GT
`attributes/curvature_values.npy` is float32, so `np.percentile` returned
`np.float32` curvature thresholds that `json.dump` refused.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.config import Config
from src.pipeline.aggregate import (
    aggregate_curvature,
    aggregate_global,
    discover_methods,
    save_aggregated_results,
)


def _make_config(tmp_path: Path, methods=None, exclude=None) -> Config:
    return Config(
        dataset={
            "name": "t",
            "objects": ["obj1", "obj2"],
            "methods": methods or [],
            "exclude_methods": exclude or [],
        },
        paths={
            "data_root": str(tmp_path / "data" / "{object}"),
            "eval_root": str(tmp_path / "eval" / "{object}"),
            "gt_artifacts_root": str(tmp_path / "gt" / "{object}"),
        },
        evaluation={"max_dist": 4.0, "fscore_thresholds": [0.5, 1.0]},
    )


def _make_combo(config: Config, obj: str, method: str, n: int = 50, seed: int = 0) -> None:
    """Create a minimal evaluated combo: results_raw mesh + metrics.json + distances + curvature zone."""
    rng = np.random.default_rng(seed)
    method_dir = config.get_method_dir(obj, method)
    (method_dir / "results_raw").mkdir(parents=True)
    (method_dir / "results_raw" / "mesh.ply").write_text("ply\n")
    eval_dir = config.get_eval_dir(obj, method)
    (eval_dir / "distances").mkdir(parents=True)
    (eval_dir / "metrics.json").write_text("{}")
    gt2data = rng.uniform(0, 2, n).astype(np.float32)
    data2gt = rng.uniform(0, 2, n).astype(np.float32)
    np.save(eval_dir / "distances" / "gt2data_dist.npy", gt2data)
    np.save(eval_dir / "distances" / "data2gt_dist.npy", data2gt)
    zones = eval_dir / "zones" / "curvature"
    zones.mkdir(parents=True)
    np.save(zones / "curvature_values.npy", rng.normal(0, 1, n).astype(np.float32))
    np.save(zones / "gt2data_dist.npy", gt2data)
    np.save(zones / "data2gt_dist.npy", data2gt)
    np.save(zones / "data2gt_curvature.npy", rng.normal(0, 1, n).astype(np.float32))


def _make_gt_curvature(config: Config, obj: str, n: int = 50, seed: int = 1) -> None:
    attrs = config.get_gt_artifacts_dir(obj) / "attributes"
    attrs.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    np.save(attrs / "curvature_values.npy", rng.normal(0, 1, n).astype(np.float32))


def test_save_aggregated_results_converts_numpy_types(tmp_path):
    """Every aggregated file must be written even when values are numpy scalars/arrays."""
    config = _make_config(tmp_path)
    payload = {
        "m": {
            "chamfer": np.float32(0.25),
            "n_gt_points": np.int64(7),
            "fscore": np.array([0.5, 0.75], dtype=np.float32),
            "nested": [np.float64(1.5), (np.int32(2), np.bool_(True))],
            "nan": float("nan"),
        },
        "thresholds": {"concave": np.float32(-0.5), "convex": np.float32(0.5)},
    }
    save_aggregated_results(
        config,
        global_metrics=payload,
        visibility_metrics=payload,
        curvature_metrics=payload,
        challenge_metrics=payload,
    )
    out = config.paths.output_root / "aggregated"
    for name in ("global", "visibility", "curvature", "challenge"):
        data = json.loads((out / f"{name}_metrics.json").read_text())
        m = data["m"]
        assert m["chamfer"] == pytest.approx(0.25)
        assert m["n_gt_points"] == 7 and isinstance(m["n_gt_points"], int)
        assert m["fscore"] == pytest.approx([0.5, 0.75])
        assert m["nested"] == [1.5, [2, True]]
        assert np.isnan(m["nan"])
        assert data["thresholds"] == pytest.approx({"concave": -0.5, "convex": 0.5})


def test_curvature_thresholds_from_float32_gt_are_builtin_floats(tmp_path):
    """Percentile thresholds computed from float32 GT curvature must be plain floats."""
    config = _make_config(tmp_path, methods=["m1"])
    _make_gt_curvature(config, "obj1")
    _make_combo(config, "obj1", "m1")
    result = aggregate_curvature(config)
    for key in ("concave", "convex"):
        assert type(result["thresholds"][key]) is float
    # End to end: the exact crash of job 120551.
    save_aggregated_results(config, curvature_metrics=result)
    saved = json.loads((config.paths.output_root / "aggregated" / "curvature_metrics.json").read_text())
    assert set(saved["m1"]) == {"concave", "flat", "convex"}


def test_discover_methods_mirrors_watcher_selection(tmp_path):
    """Empty `dataset.methods` -> evaluated methods on disk, minus exclude_methods, sorted, deduplicated."""
    config = _make_config(tmp_path, exclude=["__bak", "gomvs"])
    _make_combo(config, "obj1", "colmap_d4")
    _make_combo(config, "obj2", "colmap_d4", seed=3)
    _make_combo(config, "obj2", "neus2_d8", seed=4)
    _make_combo(config, "obj1", "neus2_d8__bak", seed=5)   # excluded backup
    _make_combo(config, "obj1", "gomvs", seed=6)          # excluded method
    # Not evaluated (no metrics.json) -> not aggregated.
    (config.get_method_dir("obj1", "pgsr_d4") / "results_raw").mkdir(parents=True)
    # GT workspace is never a method.
    (config.get_eval_root("obj1") / "Groundtruth" / "results_raw").mkdir(parents=True)
    assert discover_methods(config) == ["colmap_d4", "neus2_d8"]


def test_aggregate_global_uses_discovered_methods_when_config_lists_none(tmp_path):
    """The martine configs set `methods: []`; aggregation must not silently produce nothing."""
    config = _make_config(tmp_path)
    _make_combo(config, "obj1", "colmap_d4")
    _make_combo(config, "obj2", "colmap_d4", seed=3)
    result = aggregate_global(config)
    assert list(result) == ["colmap_d4"]
    assert result["colmap_d4"]["n_objects"] == 2
    assert result["colmap_d4"]["n_gt_points"] == 100


def test_explicit_config_methods_still_win(tmp_path):
    config = _make_config(tmp_path, methods=["colmap_d4"])
    _make_combo(config, "obj1", "colmap_d4")
    _make_combo(config, "obj1", "neus2_d8", seed=4)
    assert list(aggregate_global(config)) == ["colmap_d4"]
