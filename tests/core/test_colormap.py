"""Tests for colormap module."""

import numpy as np
import trimesh
import pytest


def test_color_mesh_by_metric_sets_vertex_colors():
    """color_mesh_by_metric should set vertex colors based on values."""
    from src.core.colormap import color_mesh_by_metric

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    values = np.array([0.0, 0.5, 1.0])

    colored = color_mesh_by_metric(mesh, values, cmap="jet", vmin=0.0, vmax=1.0)

    assert colored.visual.vertex_colors is not None
    assert colored.visual.vertex_colors.shape == (3, 4)  # RGBA
    # First vertex (value=0) should be blue-ish in jet
    # Last vertex (value=1) should be red-ish in jet
    assert colored.visual.vertex_colors[0, 0] < colored.visual.vertex_colors[2, 0]  # R increases


def test_color_mesh_by_metric_auto_range():
    """color_mesh_by_metric should auto-compute vmin/vmax if not provided."""
    from src.core.colormap import color_mesh_by_metric

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    values = np.array([10.0, 20.0, 30.0])

    colored = color_mesh_by_metric(mesh, values, cmap="viridis")

    assert colored.visual.vertex_colors is not None


def test_color_uniform():
    """color_uniform should apply uniform color to all vertices."""
    from src.core.colormap import color_uniform

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    colored = color_uniform(mesh, color=(200, 200, 200))

    assert colored.visual.vertex_colors is not None
    # All vertices should have same color
    assert np.all(colored.visual.vertex_colors[:, :3] == 200)


def test_color_by_accuracy():
    """color_by_accuracy should return colored mesh and colorbar params."""
    from src.core.colormap import color_by_accuracy

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    dist_data2gt = np.array([0.1, 0.5, 2.0])

    colored, params = color_by_accuracy(mesh, dist_data2gt, max_dist=5.0, cmap="jet")

    assert colored.visual.vertex_colors is not None
    assert params["vmin"] == 0.0
    assert params["vmax"] == 5.0
    assert params["cmap"] == "jet"
    assert "label" in params
