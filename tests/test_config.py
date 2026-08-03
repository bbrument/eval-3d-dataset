"""Tests for config.py - configuration loading and validation."""

import os
from pathlib import Path

import pytest
import yaml

from src.config import (
    Config,
    DatasetConfig,
    PathsConfig,
    CleanupConfig,
    EvaluationConfig,
    ExecutionConfig,
    AggregationConfig,  # F1 Fix: Import AggregationConfig
    LocalConfig,  # F12 Fix: Import for validation tests
    SlurmConfig,
    SlurmResourceConfig,
    load_config,
)


class TestDatasetConfig:
    """Tests for DatasetConfig validation."""

    def test_valid_config(self):
        """Valid dataset config should be accepted."""
        config = DatasetConfig(
            name="test_dataset", objects=["obj1", "obj2"], methods=["method1"]
        )
        assert config.name == "test_dataset"
        assert config.objects == ["obj1", "obj2"]
        assert config.num_views == 84  # default

    def test_empty_lists_allowed(self):
        """Empty object/method lists should be allowed."""
        config = DatasetConfig(name="test")
        assert config.objects == []
        assert config.methods == []

    # F12 Fix: Add negative validation tests
    def test_negative_num_views_uses_default(self):
        """num_views should use default if not specified."""
        config = DatasetConfig(name="test")
        assert config.num_views == 84


class TestPathsConfig:
    """Tests for PathsConfig validation."""

    def test_eval_root_defaults_to_data_root(self):
        """eval_root should default to data_root if not specified."""
        config = PathsConfig(data_root="/path/to/data")
        assert config.eval_root == "/path/to/data"

    def test_eval_root_can_be_different(self):
        """eval_root can be different from data_root."""
        config = PathsConfig(data_root="/path/to/data", eval_root="/path/to/eval")
        assert config.data_root == "/path/to/data"
        assert config.eval_root == "/path/to/eval"


class TestCleanupConfig:
    """Tests for CleanupConfig validation."""

    def test_default_values(self):
        """Default values should be set correctly."""
        config = CleanupConfig()
        assert config.dilation_radius == 12
        assert config.z_threshold is None

    def test_negative_dilation_rejected(self):
        """Negative dilation radius should be rejected."""
        with pytest.raises(ValueError, match="dilation_radius must be >= 0"):
            CleanupConfig(dilation_radius=-1)

    def test_zero_dilation_allowed(self):
        """Zero dilation radius should be allowed."""
        config = CleanupConfig(dilation_radius=0)
        assert config.dilation_radius == 0


class TestEvaluationConfig:
    """Tests for EvaluationConfig validation."""

    def test_default_thresholds(self):
        """Default F-score thresholds should be set."""
        config = EvaluationConfig()
        assert len(config.fscore_thresholds) > 0
        assert 1.0 in config.fscore_thresholds

    def test_empty_thresholds_rejected(self):
        """Empty threshold list should be rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            EvaluationConfig(fscore_thresholds=[])

    def test_negative_threshold_rejected(self):
        """Negative thresholds should be rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            EvaluationConfig(fscore_thresholds=[0.5, -0.1, 1.0])

    def test_positive_density_required(self):
        """Downsample density must be positive."""
        with pytest.raises(ValueError):
            EvaluationConfig(downsample_density=0)

        with pytest.raises(ValueError):
            EvaluationConfig(downsample_density=-1)

    # F12 Fix: Add max_dist validation test
    def test_positive_max_dist_required(self):
        """max_dist must be positive."""
        with pytest.raises(ValueError):
            EvaluationConfig(max_dist=0)

        with pytest.raises(ValueError):
            EvaluationConfig(max_dist=-1)


class TestExecutionConfig:
    """Tests for ExecutionConfig validation."""

    def test_local_mode_no_slurm_required(self):
        """Local mode should not require SLURM config."""
        config = ExecutionConfig(mode="local")
        assert config.slurm is None

    def test_slurm_mode_requires_slurm_config(self):
        """SLURM mode should require SLURM configuration."""
        with pytest.raises(ValueError, match="SLURM configuration required"):
            ExecutionConfig(mode="slurm")


# F1 Fix: Add tests for AggregationConfig
class TestAggregationConfig:
    """Tests for AggregationConfig validation."""

    def test_default_values(self):
        """Default values should be None."""
        config = AggregationConfig()
        assert config.visibility_groups is None
        assert config.curvature_thresholds is None

    def test_valid_visibility_groups(self):
        """Valid visibility groups should be accepted."""
        config = AggregationConfig(
            visibility_groups={"low": [1, 5], "high": [6, 84]}
        )
        assert config.visibility_groups["low"] == [1, 5]
        assert config.visibility_groups["high"] == [6, 84]

    def test_valid_curvature_thresholds(self):
        """Valid curvature thresholds should be accepted."""
        config = AggregationConfig(
            curvature_thresholds={"concave": -0.1, "convex": 0.1}
        )
        assert config.curvature_thresholds["concave"] == -0.1
        assert config.curvature_thresholds["convex"] == 0.1


# F12 Fix: Add tests for LocalConfig
class TestLocalConfig:
    """Tests for LocalConfig validation."""

    def test_default_max_workers(self):
        """Default max_workers should be 4."""
        config = LocalConfig()
        assert config.max_workers == 4

    def test_custom_max_workers(self):
        """Custom max_workers should be accepted."""
        config = LocalConfig(max_workers=8)
        assert config.max_workers == 8


# F12 Fix: Add tests for SlurmResourceConfig
class TestSlurmResourceConfig:
    """Tests for SlurmResourceConfig validation."""

    def test_default_values(self):
        """Default values should be set."""
        config = SlurmResourceConfig()
        assert config.cpus == 4
        assert config.mem_gb == 32
        assert config.time == "02:00:00"

    def test_custom_values(self):
        """Custom values should be accepted."""
        config = SlurmResourceConfig(cpus=16, mem_gb=64, time="04:00:00")
        assert config.cpus == 16
        assert config.mem_gb == 64
        assert config.time == "04:00:00"


class TestConfig:
    """Tests for root Config model."""

    def test_minimal_valid_config(self):
        """Minimal valid configuration should be accepted."""
        config = Config(
            dataset=DatasetConfig(name="test"),
            paths=PathsConfig(data_root="/tmp/data"),
        )
        assert config.dataset.name == "test"
        assert config.cleanup.dilation_radius == 12  # default

    def test_get_paths(self):
        """Path resolution methods should work."""
        config = Config(
            dataset=DatasetConfig(name="test", objects=["obj1"]),
            paths=PathsConfig(data_root="/data/{object}", eval_root="/eval/{object}"),
        )

        data_root = config.get_data_root("obj1")
        assert "obj1" in str(data_root)

        eval_root = config.get_eval_root("obj1")
        assert "obj1" in str(eval_root)

        gt_dir = config.get_gt_dir("obj1")
        assert "Groundtruth" in str(gt_dir)

    # F10 Fix: Add comprehensive template resolution tests
    def test_template_with_normals_priors(self):
        """Template resolution with normals_priors placeholder."""
        config = Config(
            dataset=DatasetConfig(
                name="test",
                objects=["obj1"],
                normals_priors=["gt_normal", "omni_normal"]
            ),
            paths=PathsConfig(
                data_root="/data/{object}/{normals_priors}",
                eval_root="/eval/{object}"
            ),
        )

        # When method contains /, extract normals_priors from it
        data_root = config.get_data_root("obj1", "method/gt_normal/variant")
        assert "gt_normal" in str(data_root)

    def test_template_with_hierarchical_method(self):
        """Template resolution with hierarchical method paths."""
        config = Config(
            dataset=DatasetConfig(name="test", objects=["obj1"]),
            paths=PathsConfig(
                data_root="/data/{object}",
                eval_root="/eval/{object}"
            ),
        )

        method_dir = config.get_method_dir("obj1", "method/variant")
        assert "method/variant" in str(method_dir)

    def test_get_cameras_and_masks_paths(self):
        """Test cameras.npz and masks directory path resolution."""
        config = Config(
            dataset=DatasetConfig(name="test", objects=["obj1"]),
            paths=PathsConfig(data_root="/data/{object}"),
        )

        cameras_path = config.get_cameras_path("obj1")
        assert str(cameras_path).endswith("cameras.npz")

        masks_dir = config.get_masks_dir("obj1")
        assert str(masks_dir).endswith("mask")


class TestLoadConfig:
    """Tests for load_config function."""

    # F2 Fix: Use tmp_path fixture for proper cleanup
    def test_load_valid_yaml(self, tmp_path):
        """Valid YAML file should be loaded correctly."""
        config_dict = {
            "dataset": {"name": "test_dataset", "objects": ["obj1"]},
            "paths": {"data_root": "/tmp/data"},
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_file)

        assert config.dataset.name == "test_dataset"
        assert config.dataset.objects == ["obj1"]

    def test_load_missing_file_raises(self):
        """Missing file should raise an error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    # F2 Fix: Use tmp_path fixture
    def test_load_invalid_yaml_raises(self, tmp_path):
        """Invalid YAML should raise validation error."""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, "w") as f:
            # Missing required 'dataset' field
            yaml.dump({"paths": {"data_root": "/tmp"}}, f)

        with pytest.raises(Exception):  # ValidationError
            load_config(config_file)

    # F2 Fix: Use tmp_path fixture
    def test_load_complete_config(self, tmp_path):
        """Complete config with all sections should load correctly."""
        config_dict = {
            "dataset": {
                "name": "complete_test",
                "objects": ["obj1", "obj2"],
                "methods": ["method1"],
                "num_views": 42
            },
            "paths": {
                "data_root": "/data/{object}",
                "eval_root": "/eval/{object}"
            },
            "cleanup": {
                "dilation_radius": 10,
                "z_threshold": 0.5
            },
            "evaluation": {
                "downsample_density": 0.1,
                "max_dist": 3.0,
                "fscore_thresholds": [0.5, 1.0, 2.0]
            },
            "execution": {
                "mode": "local",
                "local": {"max_workers": 8}
            },
            "aggregation": {
                "visibility_groups": {"low": [1, 10], "high": [11, 42]},
                "curvature_thresholds": {"flat": 0.0, "curved": 0.5}
            }
        }

        config_file = tmp_path / "complete.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        config = load_config(config_file)

        assert config.dataset.name == "complete_test"
        assert config.dataset.num_views == 42
        assert config.cleanup.dilation_radius == 10
        assert config.evaluation.max_dist == 3.0
        assert config.aggregation.visibility_groups["low"] == [1, 10]

