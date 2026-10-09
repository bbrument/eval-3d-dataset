"""Tests for the `visualize` CLI exit status and the viz SLURM submission.

Regression for the 2026-10-07 viz pass: 1125 `viz_*` jobs printed
"Error: pyrender not installed" yet exited 0, so SLURM reported COMPLETED
and nothing flagged the empty `visualizations/` dirs.
"""

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import src.core.rendering as rendering
from src.cli import main
from src.config import load_config


def _write_config(tmp_path: Path) -> Path:
    cfg = {
        "dataset": {"name": "t", "objects": ["obj1"], "methods": []},
        "paths": {
            "data_root": str(tmp_path / "data" / "{object}"),
            "eval_root": str(tmp_path / "eval" / "{object}"),
        },
        "execution": {
            "mode": "slurm",
            "slurm": {
                "account": "acc",
                "partition": "part",
                "cleanup": {"cpus": 1, "mem_gb": 1, "time": "00:10:00"},
                "eval": {"cpus": 1, "mem_gb": 1, "time": "00:10:00"},
                "viz": {"cpus": 1, "mem_gb": 1, "time": "00:10:00"},
            },
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_visualize_exits_nonzero_when_pyrender_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(rendering, "PYRENDER_AVAILABLE", False)
    monkeypatch.setattr(rendering, "PYRENDER_IMPORT_ERROR", "No module named 'pyrender'", raising=False)
    cfg = _write_config(tmp_path)
    result = CliRunner().invoke(main, ["-c", str(cfg), "visualize", "--object", "obj1", "--method", "m"])
    assert result.exit_code != 0
    assert "pyrender" in result.output
    # The import error itself is surfaced (a missing libOSMesa also lands here).
    assert "No module named 'pyrender'" in result.output
    # An environment problem must not lock the combo as a per-method failure.
    assert not (tmp_path / "eval" / "obj1" / "m" / ".viz_failed").exists()


def test_visualize_missing_module_fails_without_locking(tmp_path, monkeypatch):
    """A dependency missing mid-run (e.g. fast_simplification for decimation) is an environment
    error: non-zero exit, but no .viz_failed lock (it would block every combo until --force)."""
    import src.pipeline.visualize as viz

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'fast_simplification'")

    monkeypatch.setattr(rendering, "PYRENDER_AVAILABLE", True)
    monkeypatch.setattr(viz, "visualize_method", missing)
    cfg = _write_config(tmp_path)
    result = CliRunner().invoke(main, ["-c", str(cfg), "visualize", "--object", "obj1", "--method", "m"])
    assert result.exit_code != 0
    assert "fast_simplification" in result.output
    assert not (tmp_path / "eval" / "obj1" / "m" / ".viz_failed").exists()


def test_visualize_missing_required_file_fails_and_locks(tmp_path, monkeypatch):
    """A missing required input must fail its SLURM job, not report success."""
    import src.pipeline.visualize as viz

    def missing(*args, **kwargs):
        raise FileNotFoundError("required camera file missing")

    monkeypatch.setattr(rendering, "PYRENDER_AVAILABLE", True)
    monkeypatch.setattr(viz, "visualize_method", missing)
    cfg = _write_config(tmp_path)
    result = CliRunner().invoke(main, ["-c", str(cfg), "visualize", "--object", "obj1", "--method", "m"])
    assert result.exit_code != 0
    assert "required camera file missing" in result.output
    lock = tmp_path / "eval" / "obj1" / "m" / ".viz_failed"
    assert lock.exists()
    assert "required camera file missing" in lock.read_text()


def test_gt_renders_go_to_workspace_not_published_kit(tmp_path, monkeypatch):
    """GT renders + bbox.json are DERIVED: they belong in eval_root/<obj>/Groundtruth, never in the
    (writable-by-us) published gt/ kit that gt_root/gt_artifacts_root point to."""
    import numpy as np
    import trimesh
    import src.pipeline.visualize as viz
    from src.config import Config

    pub_gt = tmp_path / "pub" / "obj1" / "gt"
    pub_gt.mkdir(parents=True)
    trimesh.creation.box().export(pub_gt / "gt.ply")
    (pub_gt / "eval_cameras.sfm").write_text("{}")
    config = Config(
        dataset={"name": "t", "objects": ["obj1"]},
        paths={
            "data_root": str(tmp_path / "data" / "{object}"),
            "eval_root": str(tmp_path / "eval" / "{object}"),
            "gt_root": str(tmp_path / "pub" / "{object}" / "gt"),
            "gt_artifacts_root": str(tmp_path / "pub" / "{object}" / "gt"),
            "eval_cameras": str(tmp_path / "pub" / "{object}" / "gt" / "eval_cameras.sfm"),
        },
    )
    config.get_method_dir("obj1", "m").mkdir(parents=True)

    def fake_render_views(mesh, cameras, view_indices, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        png = output_dir / "view_000.png"
        png.write_bytes(b"png")
        return [png], {0: (0, 0, 10, 10)}

    monkeypatch.setattr(viz, "check_pyrender", lambda: None)
    monkeypatch.setattr(viz, "load_cameras_auto", lambda p: {"K": np.eye(3)[None]})
    monkeypatch.setattr(viz, "render_views", fake_render_views)

    viz.visualize_method(config, "obj1", "m", metrics=["uniform"], view_indices=[0], crop=True)

    workspace_vis = config.get_gt_dir("obj1") / "visualizations"
    assert json.loads((workspace_vis / "bbox.json").read_text()) == {"0": [0, 0, 10, 10]}
    assert (workspace_vis / "uniform" / "view_000.png").exists()
    assert not (pub_gt / "visualizations").exists()
    assert not list(workspace_vis.glob("*.tmp*"))  # bbox.json written atomically


def test_viz_jobs_are_submitted_at_lowest_priority(tmp_path, monkeypatch):
    from src.runner.slurm import SlurmRunner
    from src.watcher import scanner

    monkeypatch.setattr(SlurmRunner, "get_active_job_names", lambda self: set())
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="4242\n", stderr="")

    monkeypatch.setattr(scanner.subprocess, "run", fake_run)
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)
    submitter = scanner.SlurmSubmitter(config, cfg_path, venv_path=tmp_path / "venv")
    job = scanner.EvalJob(object_name="obj1", method_name="m", mesh_path="x.ply")
    assert submitter.submit_visualize(job) == "4242"
    assert "--nice=2147483645" in calls[0]
