"""Normal map I/O and Mean Angular Error computation."""

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import sys
sys.path.insert(0, '/home/babrument/dev/pyalicevisionlib/src')

from pathlib import Path

import cv2
import numpy as np

try:
    from pyalicevisionlib.sfmdata import load_sfmdata
    from pyalicevisionlib.mesh import load_mesh as pyav_load_mesh
    from pyalicevisionlib.rendering import render_normal_map
    HAS_PYALICEVISIONLIB = True
except ImportError:
    HAS_PYALICEVISIONLIB = False


def load_normal_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a 16-bit PNG normal map and return float normals + mask.

    Args:
        path: Path to 16-bit RGB PNG normal map.

    Returns:
        Tuple of:
            - normals: (H, W, 3) float32, unit vectors in [-1, 1]
            - mask: (H, W) bool, True where pixel is valid
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read normal map: {path}")

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    normals = img.astype(np.float32) / 65535.0 * 2.0 - 1.0
    mask = np.any(img > 0, axis=2)

    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normals = normals / norms
    normals[~mask] = 0.0

    return normals, mask


def compute_mae(
    normals_gt: np.ndarray,
    normals_pred: np.ndarray,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
) -> dict:
    """Compute Mean Angular Error between two normal maps.

    Args:
        normals_gt: (H, W, 3) float32, GT unit normals.
        normals_pred: (H, W, 3) float32, predicted unit normals.
        mask_gt: (H, W) bool, valid pixels in GT.
        mask_pred: (H, W) bool, valid pixels in prediction.

    Returns:
        Dictionary with mae_mean, mae_median, pct_below_5/10/20, n_valid_pixels, angular_error_map.
    """
    h, w = normals_gt.shape[:2]
    valid = mask_gt & mask_pred
    angular_error_map = np.zeros((h, w), dtype=np.float32)

    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "mae_mean": 0.0, "mae_median": 0.0,
            "pct_below_5": 0.0, "pct_below_10": 0.0, "pct_below_20": 0.0,
            "n_valid_pixels": 0, "angular_error_map": angular_error_map,
        }

    gt_valid = normals_gt[valid]
    pred_valid = normals_pred[valid]

    dots = np.sum(gt_valid * pred_valid, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    errors_deg = np.degrees(np.arccos(dots))

    angular_error_map[valid] = errors_deg

    return {
        "mae_mean": float(np.mean(errors_deg)),
        "mae_median": float(np.median(errors_deg)),
        "pct_below_5": float(np.mean(errors_deg < 5.0)),
        "pct_below_10": float(np.mean(errors_deg < 10.0)),
        "pct_below_20": float(np.mean(errors_deg < 20.0)),
        "n_valid_pixels": n_valid,
        "angular_error_map": angular_error_map,
    }


def render_normals_from_mesh(
    mesh,
    sfm_path: Path,
    output_dir: Path,
    chunk_size: int = 1_000_000,
    samples: int = 3,
    force: bool = False,
) -> list[Path]:
    """Render normal maps from a mesh for all views in an SfM file.

    Args:
        mesh: trimesh.Trimesh object.
        sfm_path: Path to sfm.json.
        output_dir: Output directory (will contain normals/ and masks/ subdirs).
        chunk_size: Rays per batch for ray casting.
        samples: AA samples per axis.
        force: Overwrite existing files.

    Returns:
        List of paths to newly rendered normal maps.
    """
    if not HAS_PYALICEVISIONLIB:
        raise ImportError(
            "pyalicevisionlib is required for normal map rendering. "
            "Ensure /home/babrument/dev/pyalicevisionlib/src is in PYTHONPATH."
        )

    from tqdm import tqdm

    sfm = load_sfmdata(str(sfm_path))
    cameras = sfm.get_cameras()

    normals_dir = output_dir / "normals"
    masks_dir = output_dir / "masks"
    normals_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rendered = []
    for cam in tqdm(cameras, desc="Rendering normals"):
        normal_path = normals_dir / f"{cam.view_id}.png"
        mask_path = masks_dir / f"{cam.view_id}.png"

        if normal_path.exists() and mask_path.exists() and not force:
            continue

        normal_map, mask = render_normal_map(mesh, cam, chunk_size=chunk_size, samples=samples)

        cv2.imwrite(str(normal_path), cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        rendered.append(normal_path)

    return rendered
