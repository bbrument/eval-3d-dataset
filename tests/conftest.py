"""Pytest configuration for eval_pipeline tests."""

import sys
from pathlib import Path

# F4 Fix: Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """Skip `@pytest.mark.gl` tests when pyrender has no usable OpenGL backend."""
    gl_items = [item for item in items if "gl" in item.keywords]
    if not gl_items:
        return
    from eval3d.core.rendering import PYRENDER_AVAILABLE, PYRENDER_IMPORT_ERROR

    if PYRENDER_AVAILABLE:
        return
    skip = pytest.mark.skip(reason=f"pyrender/OpenGL unavailable ({PYRENDER_IMPORT_ERROR})")
    for item in gl_items:
        item.add_marker(skip)
