"""The README's first run: `run` on the bundled lego fixture with config/example.yaml, unedited."""

import json
import math
import shutil
from pathlib import Path

import yaml
from click.testing import CliRunner

from eval3d.cli import main

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "example.yaml"
FIXTURE = REPO / "tests" / "data" / "lego"


def test_example_config_points_to_the_fixture(monkeypatch):
    from eval3d.config import load_config

    monkeypatch.chdir(REPO)
    config = load_config(CONFIG)
    assert config.dataset.objects == ["lego"]
    assert config.dataset.methods == ["pgsr_d4"]
    assert (config.get_gt_artifacts_dir("lego") / "gt_pcd.npy").is_file()
    assert config.get_cameras_path("lego").is_file()
    assert any(config.get_masks_dir("lego").glob("*.png"))


def test_run_on_the_fixture(tmp_path, monkeypatch):
    # Same steps as the README, with the workspace moved to tmp_path.
    monkeypatch.chdir(REPO)
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["paths"]["eval_root"] = str(tmp_path / "runs" / "{object}")
    cfg_path = tmp_path / "example.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    raw = tmp_path / "runs" / "lego" / "pgsr_d4" / "results_raw"
    raw.mkdir(parents=True)
    shutil.copy(FIXTURE / "pgsr_d4.ply", raw / "mesh.ply")

    result = CliRunner().invoke(main, ["-c", str(cfg_path), "run"])
    assert result.exit_code == 0, result.output
    assert "Completed: 2/2 jobs succeeded" in result.output

    metrics = json.loads((tmp_path / "runs" / "lego" / "pgsr_d4" / "eval_results" / "metrics.json").read_text())
    assert math.isfinite(metrics["chamfer"]) and 0 < metrics["chamfer"] < 1.5
    f1 = metrics["fscore"][metrics["thresholds"].index(1.0)]
    assert 0.6 < f1 < 1.0
