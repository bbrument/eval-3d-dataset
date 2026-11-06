"""`eval3d.core.rendering` must import on a machine whose GL backend is broken.

PyOpenGL without a usable GL library raises `AttributeError` (not `ImportError`) while
pyrender is imported; that used to abort pytest collection of every test importing the
module. The module must instead record the reason and report pyrender as unavailable.
"""

import importlib
import importlib.abc
import sys

import pytest


class _BrokenGLFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "pyrender" or name.startswith("pyrender."):
            raise AttributeError("'NoneType' object has no attribute 'glGetError'")
        return None


@pytest.fixture
def broken_gl(monkeypatch):
    import eval3d.core.rendering as rendering

    for name in [m for m in sys.modules if m == "pyrender" or m.startswith("pyrender.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [_BrokenGLFinder(), *sys.meta_path])
    yield rendering
    monkeypatch.undo()
    importlib.reload(rendering)


def test_import_survives_broken_gl_backend(broken_gl):
    rendering = importlib.reload(broken_gl)
    assert rendering.PYRENDER_AVAILABLE is False
    assert "AttributeError" in rendering.PYRENDER_IMPORT_ERROR


def test_check_pyrender_reports_the_reason(broken_gl):
    rendering = importlib.reload(broken_gl)
    with pytest.raises(ImportError, match="glGetError"):
        rendering.check_pyrender()
