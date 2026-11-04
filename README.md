<div align="center">
<h1>eval-3d</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/NeurIPS%202026-Paper-red.svg)](https://martine-benchmark.github.io/)
[![Dataset DOI](https://img.shields.io/badge/DOI-10.7910%2FDVN%2F2BEPEH-blue.svg)](https://doi.org/10.7910/DVN/2BEPEH)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![tests](https://github.com/martine-benchmark/eval-3d/actions/workflows/tests.yml/badge.svg)](https://github.com/martine-benchmark/eval-3d/actions/workflows/tests.yml)

</div>

**eval-3d** scores reconstructed meshes against a ground-truth mesh: Chamfer distance, precision / recall /
F-score at fixed thresholds, dense PR curves, and a `coverage` value that says how much of each cloud the numbers
were computed on. It was built for the [Martine](https://martine-benchmark.github.io/) benchmark and runs unchanged
on any dataset that follows the [input layout](docs/input-layout.md); configs for DiLiGenT-MV, LUCES-MV,
Skoltech3D, DTU and EvalMVX ship in `config/`.

## Quick start

Score one Martine object. Here the "reconstruction" is the ground truth itself, a sanity check that should give an
F-score of 1; put your own mesh (in the dataset frame, millimetres) in its place.

```bash
git clone https://github.com/martine-benchmark/eval-3d.git && cd eval-3d && uv sync
git clone https://github.com/martine-benchmark/martine-dataset.git
uv run python martine-dataset/scripts/download_data.py --groundtruth --objects rock --out data   # evaluation kit, rock only
mkdir -p runs/rock/my_method/results_raw && cp data/rock/gt/gt.ply runs/rock/my_method/results_raw/mesh.ply
sed "s|/path/to/martine_dataset|$PWD/data|; s|/path/to/workspace/runs|$PWD/runs|; s|\"methodA\"|\"my_method\"|" \
    config/martine.yaml > config/martine.local.yaml
uv run eval-pipeline -c config/martine.local.yaml cleanup -o rock    # carve to the silhouettes
uv run eval-pipeline -c config/martine.local.yaml evaluate -o rock   # Chamfer, coverage, F-score
```

QUICKSTART_RESULT

## Install

```bash
uv sync                          # core (metrics, figures); Python 3.12 from .python-version
uv sync --extra visualization    # + `visualize` (pyrender, needs an OpenGL or OSMesa library)
```

Without uv: `pip install -e ".[visualization]"` in a Python ≥ 3.10 environment. No GPU is needed. Memory is the
constraint: nearest-neighbour queries run over 10–30 M points (rock above: about 5 min and 7 GB per object and
method; the paper's Martine jobs reserved 256 GB for the largest objects).

## Usage

Every command takes `-c/--config` (required) and most take `-o/--object`, `-m/--method` and `-f/--force`.
`uv run eval-pipeline -c <config> <command> --help` lists the options.

| Command | What it does |
| :-- | :-- |
| `preprocess-gt` | Clean the GT mesh, sample its point cloud, compute per-point attributes (not needed for Martine: `gt/` ships them) |
| `cleanup` | Carve reconstructed vertices that project outside every silhouette |
| `evaluate` | Distances, Chamfer, coverage, F-score → `eval_results/metrics.json` |
| `curves` | Dense precision / recall / F-score curves from the saved distances |
| `recompute` | Rebuild `metrics.json` and curves from saved distances (new `max_dist`, new mask) |
| `run` | `cleanup` + `evaluate` over all objects and methods, locally or on SLURM (`--slurm`) |
| `apply-masks` | Per-zone distances from the GT attribute and challenge masks |
| `preprocess-challenges` | Build challenge masks from painted meshes or per-view 2D masks |
| `proxy-curvature` | GT curvature (Voronoi Covariance Measure) on a proxy mesh, tile by tile |
| `aggregate` | Collect per-cell metrics into tables across objects and methods |
| `figures` | Build a paper's figures and tables from a YAML specification |
| `latex` | LaTeX Chamfer table from the aggregated metrics |
| `visualize` | Render per-point error colourings (needs the `visualization` extra) |
| `watch`, `watch-status` | Poll for new meshes and submit evaluation jobs to SLURM; report on them |

## Documentation

| Document | Covers |
| :-- | :-- |
| [docs/input-layout.md](docs/input-layout.md) | The on-disk layout the pipeline expects, GT discovery, camera formats |
| [docs/configuration.md](docs/configuration.md) | The config keys that change your numbers; units |
| [docs/pipeline.md](docs/pipeline.md) | The end-to-end command flow, stage by stage |
| [docs/metrics.md](docs/metrics.md) | Every field of `metrics.json`; why `NaN` is a result |
| [docs/challenges.md](docs/challenges.md) | Per-region evaluation (surface properties, excluded zones), adding your own region |
| [docs/datasets.md](docs/datasets.md) | The bundled configs and how to add a dataset |
| [docs/cluster.md](docs/cluster.md) | Running on SLURM |
| [docs/figures.md](docs/figures.md) | Paper figures and tables; the Martine specification as an example |

Read `coverage` before the Chamfer distance: a low Chamfer on 20 % coverage describes a fragment, not a
reconstruction. Normal accuracy is not part of the metrics.

## Tests

```bash
uv run pytest
```

Tests that render with OpenGL are marked `gl` and skipped when no backend is available; the paper regression tests
run only when `EVAL_FIGURES_REF` points to the paper's evaluation outputs.

## Related projects

| Repository | Role |
| :-- | :-- |
| [martine-dataset](https://github.com/martine-benchmark/martine-dataset) | Martine data: download, file format, converters to method inputs |
| [eval-3d](https://github.com/martine-benchmark/eval-3d) | Mesh-vs-ground-truth evaluation (Chamfer, F-score, per-property analysis) |
| [contour-registration](https://github.com/martine-benchmark/contour-registration) | Mesh-to-image alignment by multi-view contour ICP |
| [martine-build](https://github.com/martine-benchmark/martine-build) | Dataset construction pipeline (work in progress) |

## License

MIT, see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the Martine paper:

```bibtex
@inproceedings{martine2026,
  title={Martine: Benchmarking Multi-View 3D Surface Reconstruction Across Viewpoint Coverage, Resolution, and Lighting},
  author={Bruneau, Robin and Brument, Baptiste and Giraud, Frederic and Sigrist, Bastian and Jecklin, Sascha and Fürnstahl, Philipp and Menze, Bjoern and Calvet, Lilian},
  booktitle={NeurIPS},
  year={2026}
}
```
