# Datasets and bundled configurations

[← back to README](../README.md)

The pipeline was built for a millimetre-scale 84-view benchmark (codename *martine*) and
Skoltech3D (*sk3d*), and applies unchanged to any dataset that follows the
[input layout](input-layout.md). Everything dataset-specific lives in a config file under
`config/` — `src/` carries no dataset assumptions.

## Start here — templates

| config | dataset | notes |
|---|---|---|
| `config/example_martine.yaml` | martine-style (mm, 84 views) | Relative paths, edit `paths:` and go. **Recommended starting point.** |
| `config/example_sk3d.yaml` | Skoltech3D | Relative paths; shows the metre→millimetre units handling (see [the units trap](configuration.md#the-units-trap-sk3d)). |
| `config/default.yaml` | — | Base defaults; reference for every available key. |

## Live dataset configurations

These are the configurations used in practice. They are **live configs, not templates**, and
may contain machine-specific absolute paths — copy one and repoint `paths:` before use, or
start from a template above.

| config | dataset |
|---|---|
| `config/dlmv.yaml` | DiLiGenT-MV |
| `config/dtu.yaml` | DTU |
| `config/evalmvx.yaml` | EvalMVX (Yang et al.) — 25 objects, 20 views |
| `config/lucesmv.yaml` | LUCES-MV |
| `config/sk3d.yaml` | Skoltech3D (live, `wooden_trex` and following) |
| `config/skoltech3d.yaml` | Skoltech3D (alternate variant) |

## martine / benchmark variants

The *martine* benchmark ships several task-specific variants:

| config | purpose |
|---|---|
| `config/martine_dataset.yaml` | The benchmark dataset evaluation. |
| `config/martine_refin.yaml` | Refined T2S variant (`refin_pct90`). |
| `config/martine_viz_remove.yaml` | Visualisation with `exclude_mode=remove`. |
| `config/martine_watcher.yaml` | Watcher: auto-detect all methods and objects, cleanup + eval + viz. |
| `config/martine_watcher_noautoviz.yaml` | Same watcher, without automatic visualisation. |

## Adding your own dataset

1. Arrange the data as in the [input layout](input-layout.md).
2. Copy `config/example_martine.yaml`, set `paths.eval_root` / `paths.data_root`, list your
   `dataset.objects` and `dataset.methods`.
3. Set the thresholds in **your mesh's units** — re-read
   [the units trap](configuration.md#the-units-trap-sk3d) if your meshes are not in millimetres.
4. Run the [end-to-end flow](pipeline.md).
