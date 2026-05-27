"""Tests for rendering module."""

import numpy as np
import trimesh
import pytest
from pathlib import Path


def test_offscreen_renderer_basic():
    """OffscreenRenderer should render a simple colored mesh."""
    from src.core.rendering import OffscreenRenderer, PYRENDER_AVAILABLE

    if not PYRENDER_AVAILABLE:
        pytest.skip("pyrender not available")

    # Create simple triangle
    vertices = np.array([[0, 0, 1], [1, 0, 1], [0.5, 1, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.vertex_colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8
    )

    # Simple intrinsics
    K = np.array(
        [
            [500, 0, 320, 0],
            [0, 500, 240, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Identity pose (looking at +Z)
    Rt = np.eye(4, dtype=np.float32)

    with OffscreenRenderer(640, 480, K) as renderer:
        image = renderer.render_mesh(mesh, Rt)

        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8
        # Should have some non-white pixels (the triangle)
        assert not np.all(image == 255)


def test_add_colorbar():
    """add_colorbar should add a colorbar to an image."""
    from src.core.rendering import add_colorbar

    # Create simple 100x100 white image
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    result = add_colorbar(image, vmin=0.0, vmax=10.0, cmap="jet", label="Distance [mm]")

    # Result should be wider (colorbar on right)
    assert result.shape[0] == 100  # Same height
    assert result.shape[1] > 100   # Wider
    assert result.shape[2] == 3


def test_crop_to_content():
    """crop_to_content should crop to bounding box of content."""
    from src.core.rendering import crop_to_content

    # Create 100x100 white image with small red square in center
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[40:60, 40:60] = [255, 0, 0]  # Red square

    cropped = crop_to_content(image, margin=5)

    # Should be approximately 30x30 (20x20 content + 5 margin each side)
    assert cropped.shape[0] == 30
    assert cropped.shape[1] == 30


def test_render_views(tmp_path):
    """render_views should render multiple camera views to files."""
    from src.core.rendering import render_views, PYRENDER_AVAILABLE

    if not PYRENDER_AVAILABLE:
        pytest.skip("pyrender not available")

    # Create simple mesh
    vertices = np.array([[0, 0, 1], [1, 0, 1], [0.5, 1, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.vertex_colors = np.full((3, 4), [200, 200, 200, 255], dtype=np.uint8)

    # Mock cameras dict
    K = np.array([
        [500, 0, 320, 0],
        [0, 500, 240, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    Rt = np.eye(4, dtype=np.float32)

    cameras = {
        "K": np.array([K, K]),
        "Rt": np.array([Rt, Rt]),
    }

    output_dir = tmp_path / "renders"
    paths = render_views(mesh, cameras, view_indices=[0], output_dir=output_dir, scale=1.0)

    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".png"
