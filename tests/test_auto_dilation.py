"""Tests for resolution-adaptive ('auto') mask dilation.

The auto radius scales with the linear size of the mask, calibrated so that a full-res
Martine mask (9568x6376) dilates by 48px:

    dilation = round(ref_px * sqrt(mask_n_pixels / ref_n_pixels))
"""

import pytest

from src.config import CleanupConfig, load_config
from src.core.visibility import (
    compute_auto_dilation,
    AUTO_DILATION_REF_PX,
    AUTO_DILATION_REF_N_PIXELS,
)


FULL_W, FULL_H = 9568, 6376  # Martine full resolution


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (FULL_W, FULL_H, 48),          # full resolution
        (FULL_W // 2, FULL_H // 2, 24),  # d2
        (FULL_W // 4, FULL_H // 4, 12),  # d4
        (FULL_W // 8, FULL_H // 8, 6),   # d8
    ],
)
def test_auto_dilation_reference_downscales(width, height, expected):
    """The 4 calibration points full/d2/d4/d8 must map to 48/24/12/6."""
    assert compute_auto_dilation(width * height) == expected


def test_auto_dilation_defaults_match_module_constants():
    assert AUTO_DILATION_REF_PX == 48
    assert AUTO_DILATION_REF_N_PIXELS == FULL_W * FULL_H


def test_auto_dilation_is_proportional_to_linear_size():
    """Halving the linear size halves the radius (quartering the pixel count)."""
    full = compute_auto_dilation(FULL_W * FULL_H)
    half = compute_auto_dilation((FULL_W // 2) * (FULL_H // 2))
    assert half == pytest.approx(full / 2, abs=1)


def test_auto_dilation_intermediate_resolution():
    """An intermediate size lands between the neighbouring calibration points."""
    val = compute_auto_dilation(3000 * 2000)
    assert 12 < val < 24  # between d4 and d2


def test_auto_dilation_custom_reference():
    """ref_px / ref_n_pixels are honoured."""
    # 100px at the reference res, so half the linear size -> 50px.
    assert compute_auto_dilation(1000 * 1000, ref_px=100, ref_n_pixels=2000 * 2000) == 50


@pytest.mark.parametrize("bad", [0, -5])
def test_auto_dilation_non_positive_pixels_returns_zero(bad):
    assert compute_auto_dilation(bad) == 0


def test_cleanup_config_accepts_auto():
    cfg = CleanupConfig(dilation_radius="auto")
    assert cfg.dilation_radius == "auto"
    assert cfg.dilation_ref_px == 48
    assert cfg.dilation_ref_n_pixels == FULL_W * FULL_H


def test_cleanup_config_accepts_fixed_int_backward_compat():
    cfg = CleanupConfig(dilation_radius=12)
    assert cfg.dilation_radius == 12


def test_cleanup_config_rejects_bad_string():
    with pytest.raises(ValueError):
        CleanupConfig(dilation_radius="bogus")


def test_cleanup_config_rejects_negative_int():
    with pytest.raises(ValueError):
        CleanupConfig(dilation_radius=-1)


def test_martine_dataset_config_uses_auto():
    cfg = load_config("config/martine.yaml")
    assert cfg.cleanup.dilation_radius == "auto"
    assert cfg.cleanup.dilation_ref_px == 48
    assert cfg.cleanup.dilation_ref_n_pixels == FULL_W * FULL_H
