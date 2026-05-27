"""Tests for core/normals.py — normal map I/O and MAE computation."""

import numpy as np
import pytest
import cv2
from pathlib import Path


class TestLoadNormalMap:
    def test_load_valid_normal_map(self, tmp_path):
        from src.core.normals import load_normal_map
        h, w = 10, 10
        normal_img = np.zeros((h, w, 3), dtype=np.uint16)
        normal_img[:, :, 0] = 32767
        normal_img[:, :, 1] = 32767
        normal_img[:, :, 2] = 65535
        path = tmp_path / "normals.png"
        cv2.imwrite(str(path), cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
        normals, mask = load_normal_map(path)
        assert normals.shape == (h, w, 3)
        assert mask.shape == (h, w)
        assert mask.all()
        np.testing.assert_allclose(normals[:, :, 2], 1.0, atol=1e-3)
        np.testing.assert_allclose(normals[:, :, 0], 0.0, atol=1e-3)

    def test_load_normal_map_with_background(self, tmp_path):
        from src.core.normals import load_normal_map
        h, w = 10, 10
        normal_img = np.zeros((h, w, 3), dtype=np.uint16)
        normal_img[:5, :, 0] = 32767
        normal_img[:5, :, 1] = 32767
        normal_img[:5, :, 2] = 65535
        path = tmp_path / "normals.png"
        cv2.imwrite(str(path), cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
        normals, mask = load_normal_map(path)
        assert mask[:5, :].all()
        assert not mask[5:, :].any()

    def test_normals_are_unit_length(self, tmp_path):
        from src.core.normals import load_normal_map
        h, w = 10, 10
        n = np.array([1, 1, 1], dtype=np.float64) / np.sqrt(3)
        encoded = ((n + 1.0) / 2.0 * 65535).astype(np.uint16)
        normal_img = np.zeros((h, w, 3), dtype=np.uint16)
        normal_img[:, :] = encoded
        path = tmp_path / "normals.png"
        cv2.imwrite(str(path), cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
        normals, mask = load_normal_map(path)
        lengths = np.linalg.norm(normals[mask], axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-3)


class TestComputeMae:
    def test_identical_normals(self):
        from src.core.normals import compute_mae
        h, w = 10, 10
        normals = np.zeros((h, w, 3), dtype=np.float32)
        normals[:, :, 2] = 1.0
        mask = np.ones((h, w), dtype=bool)
        result = compute_mae(normals, normals, mask, mask)
        assert result["mae_mean"] == 0.0
        assert result["mae_median"] == 0.0
        assert result["pct_below_5"] == 1.0
        assert result["n_valid_pixels"] == 100

    def test_perpendicular_normals(self):
        from src.core.normals import compute_mae
        h, w = 10, 10
        gt = np.zeros((h, w, 3), dtype=np.float32)
        gt[:, :, 2] = 1.0
        pred = np.zeros((h, w, 3), dtype=np.float32)
        pred[:, :, 0] = 1.0
        mask = np.ones((h, w), dtype=bool)
        result = compute_mae(gt, pred, mask, mask)
        np.testing.assert_allclose(result["mae_mean"], 90.0, atol=0.1)
        assert result["pct_below_5"] == 0.0

    def test_known_angle(self):
        from src.core.normals import compute_mae
        h, w = 10, 10
        gt = np.zeros((h, w, 3), dtype=np.float32)
        gt[:, :, 2] = 1.0
        pred = np.zeros((h, w, 3), dtype=np.float32)
        pred[:, :, 0] = np.sin(np.radians(45))
        pred[:, :, 2] = np.cos(np.radians(45))
        mask = np.ones((h, w), dtype=bool)
        result = compute_mae(gt, pred, mask, mask)
        np.testing.assert_allclose(result["mae_mean"], 45.0, atol=0.1)

    def test_mask_intersection(self):
        from src.core.normals import compute_mae
        h, w = 10, 10
        gt = np.zeros((h, w, 3), dtype=np.float32)
        gt[:, :, 2] = 1.0
        pred = np.zeros((h, w, 3), dtype=np.float32)
        pred[:, :, 2] = 1.0
        mask_gt = np.ones((h, w), dtype=bool)
        mask_pred = np.zeros((h, w), dtype=bool)
        mask_pred[:5, :] = True
        result = compute_mae(gt, pred, mask_gt, mask_pred)
        assert result["n_valid_pixels"] == 50

    def test_angular_error_map_shape(self):
        from src.core.normals import compute_mae
        h, w = 8, 12
        gt = np.zeros((h, w, 3), dtype=np.float32)
        gt[:, :, 2] = 1.0
        pred = gt.copy()
        mask = np.ones((h, w), dtype=bool)
        result = compute_mae(gt, pred, mask, mask)
        assert result["angular_error_map"].shape == (h, w)

    def test_empty_mask_returns_zeros(self):
        from src.core.normals import compute_mae
        h, w = 10, 10
        gt = np.zeros((h, w, 3), dtype=np.float32)
        gt[:, :, 2] = 1.0
        pred = gt.copy()
        mask = np.zeros((h, w), dtype=bool)
        result = compute_mae(gt, pred, mask, mask)
        assert result["n_valid_pixels"] == 0
        assert result["mae_mean"] == 0.0
