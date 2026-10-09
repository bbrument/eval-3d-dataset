"""SLURM job wrapping: machine-specific runtime setup comes from an optional script, never from the code."""

from pathlib import Path

from src.config import SlurmConfig
from src.watcher.scanner import wrap_command


def test_setup_script_defaults_to_none():
    assert SlurmConfig(account="acc", partition="part").setup_script is None


def test_setup_script_is_parsed():
    cfg = SlurmConfig(account="acc", partition="part", setup_script="config/setup_env.local.sh")
    assert cfg.setup_script == "config/setup_env.local.sh"


def test_wrap_without_setup_script_has_no_machine_paths():
    wrapped = wrap_command("eval-pipeline evaluate", Path("/venv"), None)
    assert "/apps/" not in wrapped and "LD_LIBRARY_PATH" not in wrapped
    assert "export PYOPENGL_PLATFORM=osmesa" in wrapped
    assert wrapped.endswith("source /venv/bin/activate && eval-pipeline evaluate")


def test_wrap_sources_setup_script_first():
    wrapped = wrap_command("eval-pipeline evaluate", Path("/venv"), "/my dir/setup_env.local.sh")
    assert wrapped.startswith("source '/my dir/setup_env.local.sh' && ")
    assert wrapped.endswith("source /venv/bin/activate && eval-pipeline evaluate")
