"""Smoke tests for Robin's mask-based watertight culling (cleanup option)."""

import numpy as np
import pytest

from src.config import CleanupConfig, Config, DatasetConfig, PathsConfig
from src.core.visibility import estimate_normal_flip, filter_by_issue_watertight


def test_config_watertight_defaults_off():
    cfg = CleanupConfig()
    assert cfg.watertight_culling is False
    assert cfg.watertight_orientation_ratio == pytest.approx(0.7)


def test_estimate_normal_flip_truth_table():
    # Mostly back-facing under un-flipped normals -> needs a flip.
    assert estimate_normal_flip(2, 10) is True
    # Mostly front-facing -> no flip.
    assert estimate_normal_flip(8, 10) is False
    # Exactly half is not "< 0.5" -> no flip.
    assert estimate_normal_flip(5, 10) is False
    # No evidence -> no flip.
    assert estimate_normal_flip(0, 0) is False


def test_filter_by_issue_watertight_empty_vertices():
    counts = filter_by_issue_watertight(
        np.zeros((0, 3), dtype=np.float32),
        mesh=None,
        cameras={"P": [], "Rt": []},
        masks_issue_watertight_dir="/nonexistent",
        show_progress=False,
    )
    assert counts.shape == (0,)


def test_get_masks_issue_watertight_dir_path():
    cfg = Config(
        dataset=DatasetConfig(name="t", objects=["obj"], methods=["m"]),
        paths=PathsConfig(
            data_root="/data/{object}",
            eval_root="/eval/{object}",
        ),
    )
    p = cfg.get_masks_issue_watertight_dir("obj")
    assert p.name == "masks_issue_watertight"
    assert p.parent == cfg.get_gt_dir("obj")
