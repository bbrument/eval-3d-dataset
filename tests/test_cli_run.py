"""`run -o/-m`: one cell instead of the config's lists."""

from pathlib import Path

from click.testing import CliRunner

from src.cli import main
from src.runner.local import LocalRunner

CONFIG = Path(__file__).resolve().parents[1] / "config" / "martine.yaml"


def _capture(monkeypatch):
    seen = {}

    def fake_run_pipeline(self, config, objects=None, methods=None, stages=None):
        seen.update(objects=objects, methods=methods, stages=stages)
        return {}

    monkeypatch.setattr(LocalRunner, "run_pipeline", fake_run_pipeline)
    return seen


def test_run_object_and_method_options(monkeypatch):
    seen = _capture(monkeypatch)
    result = CliRunner().invoke(main, ["-c", str(CONFIG), "run", "-o", "rock", "-m", "my_method"])
    assert result.exit_code == 0, result.output
    assert seen["objects"] == ["rock"] and seen["methods"] == ["my_method"]


def test_run_defaults_to_the_config_lists(monkeypatch):
    seen = _capture(monkeypatch)
    result = CliRunner().invoke(main, ["-c", str(CONFIG), "run"])
    assert result.exit_code == 0, result.output
    assert seen["objects"] is None and seen["methods"] is None
    assert seen["stages"] == ["cleanup", "evaluate"]
