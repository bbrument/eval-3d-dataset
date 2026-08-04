# eval-3d-dataset

Evaluation pipeline for multi-view 3D surface reconstruction benchmarks.

It takes a ground-truth mesh and one reconstructed mesh per method, and produces Chamfer
distance, precision/recall/F-score at fixed thresholds, dense PR curves, a `coverage` figure
telling you how much of the cloud those numbers were actually computed on, and optional metric
visualisations.

It was built for a **millimetre-scale 84-view benchmark** (codename *martine*) and for
**Skoltech3D** (*sk3d*), but nothing dataset-specific lives in `src/` — any dataset that follows
the [input layout](docs/input-layout.md) works, driven entirely by a config file. Ready-to-edit
configurations for martine, sk3d, DiLiGenT-MV, DTU, LUCES-MV and EvalMVX ship in `config/`; see
the [datasets guide](docs/datasets.md).

---

## ⚠️ Results from earlier versions are not comparable

If you evaluated with a copy of this pipeline from before **July 2026**, your numbers differ from
those produced today, and the difference is not a rounding matter. Four defects fixed in
July–August 2026 changed reported results:

1. **Chamfer returned `0.0`** — the *best* possible score — when no point survived `max_dist`, so
   total failures sorted first. It now returns `NaN`.
2. **`coverage` did not exist**, so a Chamfer computed on 1% of the points looked like a score. It
   is now reported next to every Chamfer.
3. **Exclusion masks were applied inconsistently** across the evaluate / curve / recompute paths.
   All three now share one masking module.
4. **Dense F-score curves were overwritten** by the coarse reporting grid.

Cross-version results should be regenerated, not reconciled.
[`CHANGELOG.md`](CHANGELOG.md) documents each defect in full — what was wrong, why it mattered,
and what changed in the output.

---

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
pip install -e ".[visualization]"   # only for `visualize` (needs OSMesa)
```

Python ≥ 3.10. Dependencies: numpy, trimesh, scikit-learn, OpenCV, pydantic v2, click, PyYAML.
**No GPU is required** for the metrics. The real constraint is memory: nearest-neighbour queries
run over 10–30 M points and take roughly 20 minutes per object×method cell.

Every command takes `-c/--config`; it is required.

```bash
eval-pipeline -c config/example_martine.yaml --help      # installed entry point
python -m src.cli -c config/example_martine.yaml --help  # equivalent, without installing
```

## Quick start

```bash
CFG=config/example_martine.yaml     # or config/example_sk3d.yaml
eval-pipeline -c $CFG preprocess-gt --clean-gt   # clean + sample the GT mesh
eval-pipeline -c $CFG cleanup                    # carve reconstructions to the silhouettes
eval-pipeline -c $CFG evaluate                   # Chamfer, coverage, F-score
eval-pipeline -c $CFG curves -n 100              # dense PR curves
```

Full explanation of each stage: [docs/pipeline.md](docs/pipeline.md).

---

## Documentation

| document | what it covers |
|---|---|
| [docs/input-layout.md](docs/input-layout.md) | The on-disk shape the pipeline expects, GT mesh discovery, camera formats. |
| [docs/configuration.md](docs/configuration.md) | The config keys that change your numbers, masks, and the metre/millimetre units trap. |
| [docs/pipeline.md](docs/pipeline.md) | The end-to-end command flow and every subcommand. |
| [docs/metrics.md](docs/metrics.md) | Every field in `metrics.json`, and why `NaN` is a result. |
| [docs/datasets.md](docs/datasets.md) | The bundled `config/` files, per dataset, and how to add your own. |
| [docs/cluster.md](docs/cluster.md) | Running on SLURM via the job templates. |
| [CHANGELOG.md](CHANGELOG.md) | The four cross-version result changes, in full. |

---

## Tests

```bash
python -m pytest tests/ -q
```

The four `tests/core/test_rendering.py` cases need an OpenGL/OSMesa context and fail in a
headless environment. Everything else must pass.

---

## Scope

This pipeline evaluates **geometry**: Chamfer distance, precision/recall/F-score, coverage.

It does **not** evaluate surface orientation. The normal-map / Mean-Angular-Error path was
removed in August 2026: it had never produced a reported number in any results file, and it was
the only component requiring an unpublished internal library.

---

## Licence

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Baptiste Brument.
