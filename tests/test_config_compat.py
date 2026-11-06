"""Configs written before `paths.dataset_root` existed load unchanged and resolve to the same paths."""

from pathlib import Path

import pytest
import yaml

from eval3d.config import load_config

REPO = Path(__file__).resolve().parents[1]

# Published-layout config in the format used before {dataset_root}: absolute templates, object aliases,
# normals priors, and the legacy fields left implicit.
LEGACY = """
dataset:
  name: legacy
  objects: ["01_rock"]
  methods: ["colmap_d8"]
  normals_priors: ["sdm", "uni"]
  object_aliases: {"01_rock": "rock"}
paths:
  data_root: "/data/{source_object}/{normals_priors}/mvs"
  gt_root: "/data/{source_object}/gt"
  gt_artifacts_root: "/data/{source_object}/gt"
  eval_cameras: "/data/{source_object}/gt/eval_cameras.sfm"
  eval_masks: "/data/{source_object}/gt/masks_eval"
  watertight_masks: "/data/{source_object}/gt/masks_watertight"
  eval_root: "/runs/{object}"
"""

EXPECTED = {
    ("get_data_root", ("01_rock", "colmap_d8")): "/data/rock/sdm/mvs",
    ("get_data_root", ("01_rock", "x/uni/y")): "/data/rock/uni/mvs",
    ("get_gt_source_dir", ("01_rock",)): "/data/rock/gt",
    ("get_gt_artifacts_dir", ("01_rock",)): "/data/rock/gt",
    ("get_cameras_path", ("01_rock",)): "/data/rock/gt/eval_cameras.sfm",
    ("get_masks_dir", ("01_rock",)): "/data/rock/gt/masks_eval",
    ("get_watertight_masks_dir", ("01_rock",)): "/data/rock/gt/masks_watertight",
    ("get_eval_root", ("01_rock",)): "/runs/01_rock",
    ("get_eval_dir", ("01_rock", "colmap_d8")): "/runs/01_rock/colmap_d8/eval_results",
    ("get_cleaned_mesh_path", ("01_rock", "colmap_d8")): "/runs/01_rock/colmap_d8/results_cleaned/mesh.ply",
    ("get_raw_mesh_path", ("01_rock", "colmap_d8")): "/runs/01_rock/colmap_d8/results_raw/mesh.ply",
}


@pytest.fixture
def legacy(tmp_path):
    p = tmp_path / "legacy.yaml"
    p.write_text(LEGACY)
    return load_config(p)


def test_legacy_config_loads_without_dataset_root(legacy):
    assert legacy.paths.dataset_root is None
    assert legacy.paths.output_root == Path("/runs")


@pytest.mark.parametrize("getter,args", list(EXPECTED))
def test_legacy_paths_unchanged(legacy, getter, args):
    assert str(getattr(legacy, getter)(*args)) == EXPECTED[(getter, args)]


def test_relative_template_resolves_from_cwd(tmp_path, monkeypatch):
    cfg = yaml.safe_load(LEGACY)
    cfg["paths"]["eval_root"] = "runs/{object}"
    p = tmp_path / "rel.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    assert load_config(p).get_eval_root("01_rock") == tmp_path.resolve() / "runs" / "01_rock"


@pytest.mark.parametrize("name", sorted(p.name for p in (REPO / "config").glob("*.yaml")))
def test_bundled_configs_load(name):
    config = load_config(REPO / "config" / name)
    for obj in config.dataset.objects[:1]:
        config.get_gt_artifacts_dir(obj)
        config.get_cameras_path(obj)
