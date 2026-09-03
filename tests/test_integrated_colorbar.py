"""Smoke tests for the integrated_colorbar visualization option.

The colorbar rendering lives in src/core/rendering.py, whose import pulls in pyrender /
OpenGL. On a headless machine without a usable GL backend that import raises at module
load; those tests skip rather than fail (the coloring math is backend-independent and is
exercised whenever a GL backend, e.g. PYOPENGL_PLATFORM=egl, is available).
"""

import numpy as np
import pytest

from src.config import VisualizationConfig


def test_config_integrated_colorbar_default_off():
    assert VisualizationConfig().integrated_colorbar is False


def _import_add_colorbar():
    try:
        from src.core.rendering import add_colorbar
        return add_colorbar
    except Exception as exc:  # pragma: no cover - depends on GL backend availability
        pytest.skip(f"rendering/GL backend unavailable in this environment: {exc}")


def test_add_colorbar_reserve_label_space_widens_output():
    add_colorbar = _import_add_colorbar()
    img = np.full((80, 300, 3), 255, dtype=np.uint8)

    default = add_colorbar(img, 0.0, 4.0, "viridis", "acc (mm)", position="right")
    reserved = add_colorbar(
        img, 0.0, 4.0, "viridis", "acc (mm)", position="right", reserve_label_space=True
    )

    # Both keep the image height and add width for the bar; reserve_label_space adds a
    # fixed label strip so it is at least as wide as the proportional default.
    assert default.shape[0] == 80
    assert reserved.shape[0] == 80
    assert default.shape[1] > 300
    assert reserved.shape[1] >= default.shape[1]


def test_visualize_method_importable():
    try:
        from src.pipeline.visualize import visualize_method  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"visualize import needs GL backend: {exc}")
    import inspect

    sig = inspect.signature(visualize_method)
    assert "integrated_colorbar" in sig.parameters
    assert "excluded_override_path" in sig.parameters
