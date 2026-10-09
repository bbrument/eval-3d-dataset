"""Smoke tests for Robin's mask-based watertight culling (cleanup option)."""

import numpy as np
import pytest

from src.config import CleanupConfig, Config, DatasetConfig, PathsConfig
from src.core.visibility import estimate_normal_flip, filter_by_issue_watertight


def test_config_watertight_orientation_default():
    cfg = CleanupConfig()
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


def test_get_watertight_masks_dir_path():
    cfg = Config(
        dataset=DatasetConfig(name="t", objects=["obj"], methods=["m"]),
        paths=PathsConfig(
            data_root="/data/{object}",
            eval_root="/eval/{object}",
        ),
    )
    p = cfg.get_watertight_masks_dir("obj")
    assert p.name == "masks_watertight"
    assert p.parent == cfg.get_gt_dir("obj")


def test_get_explicit_watertight_masks_dir_path():
    cfg = Config(
        dataset=DatasetConfig(name="t", objects=["01_obj"], object_aliases={"01_obj": "obj"}),
        paths=PathsConfig(
            data_root="/data/{source_object}",
            watertight_masks="/published/{source_object}/gt/masks_watertight",
        ),
    )
    assert str(cfg.get_watertight_masks_dir("01_obj")).endswith(
        "/published/obj/gt/masks_watertight"
    )
