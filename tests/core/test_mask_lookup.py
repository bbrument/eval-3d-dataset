"""Mask lookup at the pixel centre (half-pixel convention).

AliceVision cameras (SfMData, e.g. the published Martine ``gt/eval_cameras.sfm``) put the centre
of the top-left pixel at (0, 0): pixel (i, j) covers [i - 0.5, i + 0.5) x [j - 0.5, j + 0.5), so a
projected point must be looked up at the NEAREST pixel centre, and the image spans
-0.5 <= x < w - 0.5. The former lookup by truncation (``x.astype(int)``) read pixel floor(x),
i.e. the mask shifted by half a pixel. Camera files whose convention is not established
(``cameras.npz``) keep the truncation (centre of the top-left pixel at (0.5, 0.5)).
"""

import json

import numpy as np
import trimesh
from PIL import Image

import src.core.visibility as vis
from src.core.camera import load_cameras, load_cameras_auto
from src.core.visibility import (
    check_points_in_bounds,
    check_points_in_mask,
    compute_visibility_count,
    filter_by_issue_watertight,
    filter_by_visibility,
)
from src.pipeline.preprocess_challenges import _process_png

# Pixel-centre conventions (coordinate of the centre of the top-left pixel); module attributes
# looked up at test time.
ALICEVISION, TRUNCATE = 0.0, 0.5

W = H = 10
# Probes around the single masked pixel (5, 5): x (or y) = 4.4, 4.6, 5.4, 5.6.
PROBES = [(4.4, 5.0), (4.6, 5.0), (5.4, 5.0), (5.6, 5.0), (5.0, 4.4), (5.0, 4.6), (5.0, 5.4), (5.0, 5.6)]
IN_PIXEL_5_5 = [False, True, True, False, False, True, True, False]       # nearest centre is (5, 5)
TRUNCATED = [False, False, True, True, False, False, True, True]          # floor(x) == 5


def _write_sfm(path):
    """One AliceVision view of W x H pixels, fx = fy = 1, principal point at the image centre,
    identity pose: a camera-frame point (x - 5, y - 5, 1) projects to pixel coordinates (x, y)."""
    sfm = {
        "views": [{"viewId": "7", "poseId": "7", "intrinsicId": "1", "path": "000.exr"}],
        "intrinsics": [{
            "intrinsicId": "1", "width": str(W), "height": str(H),
            "focalLength": "3.6", "sensorWidth": "36", "principalPoint": ["0", "0"],
        }],
        "poses": [{"poseId": "7", "pose": {"transform": {
            "rotation": ["1", "0", "0", "0", "1", "0", "0", "0", "1"], "center": ["0", "0", "0"],
        }}}],
    }
    path.write_text(json.dumps(sfm))
    return path


def _world(pixels):
    """World points projecting to the given pixel coordinates (the loader applies diag(1, -1, -1))."""
    xy = np.asarray(pixels, dtype=float)
    return np.column_stack([xy[:, 0] - 5.0, -(xy[:, 1] - 5.0), -np.ones(len(xy))])


def _single_pixel_mask():
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[5, 5] = 255
    return mask


def _far_mesh():
    """Occluder behind the camera: never on a ray from the probes to the camera centre."""
    return trimesh.Trimesh([[100.0, 100.0, 50.0], [101.0, 100.0, 50.0], [100.0, 101.0, 50.0]], [[0, 1, 2]])


def _sfm_scene(tmp_path):
    cams = load_cameras_auto(_write_sfm(tmp_path / "cameras.sfm"))
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.fromarray(_single_pixel_mask()).save(masks / "000.png")
    return cams, masks


# --- The lookup itself -------------------------------------------------------------------------

def test_alicevision_lookup_rounds_to_the_nearest_pixel_centre():
    P = np.eye(4)
    pts = np.column_stack([np.asarray(PROBES), np.ones(len(PROBES))])
    _, in_mask = check_points_in_mask(pts, P, _single_pixel_mask() > 0, pixel_centre=vis.PIXEL_CENTRE_ALICEVISION)
    assert in_mask.tolist() == IN_PIXEL_5_5


def test_alicevision_image_bounds_lie_half_a_pixel_outside_the_centres():
    P = np.eye(4)
    xy = [(-0.6, 5), (-0.5, 5), (9.49, 5), (9.5, 5), (5, -0.5), (5, 9.5)]
    pts = np.column_stack([np.asarray(xy, float), np.ones(len(xy))])
    expected = [False, True, True, False, True, False]
    assert check_points_in_bounds(pts, P, (H, W), pixel_centre=vis.PIXEL_CENTRE_ALICEVISION).tolist() == expected
    in_bounds, in_mask = check_points_in_mask(pts, P, np.ones((H, W), bool), pixel_centre=vis.PIXEL_CENTRE_ALICEVISION)
    assert in_bounds.tolist() == expected and in_mask.tolist() == expected


def test_convention_constants():
    assert (vis.PIXEL_CENTRE_ALICEVISION, vis.PIXEL_CENTRE_TRUNCATE) == (ALICEVISION, TRUNCATE)


def test_truncation_convention_is_the_former_lookup():
    P = np.eye(4)
    pts = np.column_stack([np.asarray(PROBES), np.ones(len(PROBES))])
    _, in_mask = check_points_in_mask(pts, P, _single_pixel_mask() > 0, pixel_centre=vis.PIXEL_CENTRE_TRUNCATE)
    assert in_mask.tolist() == TRUNCATED
    xy = np.array([[-0.01, 5.0, 1.0], [0.0, 5.0, 1.0], [9.99, 5.0, 1.0], [10.0, 5.0, 1.0]])
    assert check_points_in_bounds(xy, P, (H, W), pixel_centre=vis.PIXEL_CENTRE_TRUNCATE).tolist() == [False, True, True, False]


# --- Each camera loader declares its convention -------------------------------------------------

def test_sfmdata_cameras_declare_alicevision_pixel_centres(tmp_path):
    assert load_cameras_auto(_write_sfm(tmp_path / "cameras.sfm"))["pixel_centre"] == ALICEVISION


def test_npz_cameras_keep_the_truncation_lookup(tmp_path):
    K = np.eye(4)
    np.savez(tmp_path / "cameras.npz", world_mat_0=K, camera_mat_0=K, camera_mat_inv_0=K, world_mat_inv_0=K)
    assert load_cameras(tmp_path / "cameras.npz")["pixel_centre"] == TRUNCATE


# --- Every consumer of the masks follows the cameras' convention --------------------------------

def test_visibility_count_with_alicevision_cameras(tmp_path):
    cams, masks = _sfm_scene(tmp_path)
    counts = compute_visibility_count(_world(PROBES), _far_mesh(), cams, masks, show_progress=False)
    assert counts.tolist() == [int(v) for v in IN_PIXEL_5_5]


def test_visibility_count_bounds_without_masks(tmp_path):
    cams, masks = _sfm_scene(tmp_path)
    pts = _world([(-0.4, 5.0), (9.6, 5.0)])
    counts = compute_visibility_count(pts, _far_mesh(), cams, masks, show_progress=False, use_masks=False)
    assert counts.tolist() == [1, 0]


def test_cleanup_filter_with_alicevision_cameras(tmp_path):
    cams, masks = _sfm_scene(tmp_path)
    keep = filter_by_visibility(_world(PROBES), _far_mesh(), cams, masks, dilation_radius=0, show_progress=False)
    assert keep.tolist() == IN_PIXEL_5_5


def test_watertight_culling_with_alicevision_cameras(tmp_path):
    cams, masks = _sfm_scene(tmp_path)
    # A small fan facing the camera; its first vertices are probes around pixel (5, 5).
    probes = [(4.6, 5.0), (5.6, 5.0), (5.0, 4.6), (5.0, 5.6)]
    verts = _world(probes + [(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)])
    faces = np.array([[4, 5, 6], [4, 6, 7], [0, 2, 1], [0, 1, 3]])     # consistent winding
    mesh = trimesh.Trimesh(verts, faces, process=False)
    counts = filter_by_issue_watertight(mesh.vertices, mesh, cams, masks, show_progress=False)
    assert counts[:4].tolist() == [1, 0, 1, 0]


def test_png_challenge_with_alicevision_cameras(tmp_path):
    cams, _ = _sfm_scene(tmp_path)
    cams = dict(cams, view_ids=["7"], camera_centers=np.zeros((1, 3)))
    png = tmp_path / "7_lambertian.png"
    Image.fromarray(_single_pixel_mask()).save(png)
    probes = [(4.6, 5.0), (5.6, 5.0), (5.0, 4.6), (5.0, 5.6)]
    verts = _world(probes + [(3.0, 3.0), (7.0, 3.0), (7.0, 7.0)])
    gt_mesh = trimesh.Trimesh(verts, [[4, 5, 6]], process=False)
    marked = _process_png(png, gt_mesh, verts[:4], cams, max_dist=1e-3)
    assert marked.tolist() == [True, False, True, False]
