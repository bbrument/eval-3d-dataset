"""Integration test for the MAE normal evaluation pipeline."""

import json
import numpy as np
import cv2
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def mae_test_env(tmp_path):
    """Set up a minimal environment for MAE evaluation testing."""
    objects_root = tmp_path / "objects"
    eval_root = tmp_path / "eval" / "01_rock"
    gt_dir = eval_root / "Groundtruth"
    gt_dir.mkdir(parents=True)

    # Create a simple GT mesh (flat plane)
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]]
    )
    mesh.export(str(gt_dir / "gt_10mm.ply"))

    # Create GT pcd
    np.save(gt_dir / "gt_pcd.npy", np.array(mesh.vertices))

    # Create synthetic GT normals (all pointing +Z)
    normals_dir = gt_dir / "normals" / "mvs" / "d1" / "normals"
    masks_dir = gt_dir / "normals" / "mvs" / "d1" / "masks"
    normals_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    h, w = 50, 50
    # Normal = (0, 0, 1) -> encoded as (32767, 32767, 65535) in RGB uint16
    gt_normal_img = np.zeros((h, w, 3), dtype=np.uint16)
    gt_normal_img[:, :, 0] = 32767  # R = X = 0
    gt_normal_img[:, :, 1] = 32767  # G = Y = 0
    gt_normal_img[:, :, 2] = 65535  # B = Z = 1
    # cv2 writes BGR, so convert RGB->BGR
    cv2.imwrite(str(normals_dir / "100.png"), cv2.cvtColor(gt_normal_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(masks_dir / "100.png"), np.full((h, w), 255, dtype=np.uint8))

    # Create method normals (10 degrees off from +Z toward +X)
    angle_rad = np.radians(10)
    pred_normal = np.array([np.sin(angle_rad), 0, np.cos(angle_rad)])
    encoded = ((pred_normal + 1.0) / 2.0 * 65535).astype(np.uint16)
    pred_normal_img = np.zeros((h, w, 3), dtype=np.uint16)
    pred_normal_img[:, :] = encoded

    method_normals_dir = objects_root / "01_rock" / "09_unimsps_refin_pct90" / "normals"
    method_normals_dir.mkdir(parents=True)
    cv2.imwrite(str(method_normals_dir / "100.png"), cv2.cvtColor(pred_normal_img, cv2.COLOR_RGB2BGR))

    # Config
    config_data = {
        "dataset": {"name": "test", "objects": ["01_rock"]},
        "paths": {
            "data_root": str(tmp_path / "{object}/data"),
            "eval_root": str(tmp_path / "eval/{object}"),
        },
        "normals": {
            "enabled": True,
            "objects_root": str(objects_root),
            "pose_sources": {"mvs": "{object}/08c_undisto{downscale}_refin_pct90/mvs/sfm.json"},
            "method_pose_mapping": {"default": "mvs"},
            "normal_dirs": {
                "unimsps": "{objects_root}/{object}/09_unimsps{downscale}_refin_pct90/normals",
            },
            "method_normal_source": {"test_unimsps": "unimsps"},
        },
    }

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(config_data))

    return {
        "config_path": cfg_path,
        "eval_root": eval_root,
        "tmp_path": tmp_path,
    }


def test_evaluate_normals_precomputed(mae_test_env):
    """Test MAE evaluation with pre-computed normals."""
    from src.config import load_config
    from src.pipeline.evaluate import evaluate_normals

    config = load_config(str(mae_test_env["config_path"]))
    result = evaluate_normals(config, "01_rock", "test_unimsps", force=True)

    assert result is not None
    assert "normals_mae_mean" in result
    # Expect ~10 degrees since we set the angle to 10 degrees
    np.testing.assert_allclose(result["normals_mae_mean"], 10.0, atol=1.0)
    assert result["normals_pct_below_20"] == 1.0
    assert result["normals_n_views"] == 1
    assert result["normals_pose_source"] == "mvs"

    # Check per-view error map was saved
    per_view_dir = mae_test_env["eval_root"] / "test_unimsps" / "normals_eval" / "per_view"
    assert (per_view_dir / "100.npy").exists()

    # Load and check the error map
    error_map = np.load(per_view_dir / "100.npy")
    assert error_map.shape == (50, 50)
    valid = error_map > 0
    np.testing.assert_allclose(error_map[valid], 10.0, atol=1.0)


def test_evaluate_normals_skips_when_disabled(mae_test_env):
    """Should return None when normals.enabled is False."""
    from src.config import load_config
    from src.pipeline.evaluate import evaluate_normals

    config = load_config(str(mae_test_env["config_path"]))
    config.normals.enabled = False

    result = evaluate_normals(config, "01_rock", "test_unimsps")
    assert result is None


def test_evaluate_normals_skips_when_no_gt(mae_test_env):
    """Should return None when GT normals don't exist for the pose_source/downscale."""
    from src.config import load_config
    from src.pipeline.evaluate import evaluate_normals

    config = load_config(str(mae_test_env["config_path"]))
    # This method maps to mvs/d2 which doesn't exist in test env
    result = evaluate_normals(config, "01_rock", "test_unimsps_d2", force=True)
    # test_unimsps_d2 is not in method_normal_source, so it would try mesh rendering
    # which would fail, but the GT normals for d2 don't exist so it should skip
    assert result is None
