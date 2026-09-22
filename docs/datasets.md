# Datasets and bundled configurations

[← back to README](../README.md)

The pipeline was built for a millimetre-scale 84-view benchmark (codename *martine*) and
Skoltech3D (*skoltech3d*), and applies unchanged to any dataset that follows the
[input layout](input-layout.md). Everything dataset-specific lives in a config file under
`config/` — `src/` carries no dataset assumptions.

## Bundled configurations

These are the live configs used in practice. They may contain machine-specific absolute
paths — copy one and repoint `paths:` before use. `config/default.yaml` holds the base
defaults and documents every available key.

| config | dataset |
|---|---|
| `config/martine.yaml` | Martine (mm, 84 views) — the benchmark evaluation |
| `config/martine_watcher.yaml` | Martine watcher: auto-detect methods/objects, cleanup + eval + viz (local mode) |
| `config/skoltech3d.yaml` | Skoltech3D (`wooden_trex` and following; metre→millimetre units handling — see [the units trap](configuration.md#the-units-trap-sk3d)) |
| `config/diligentmv.yaml` | DiLiGenT-MV |
| `config/dtu.yaml` | DTU |
| `config/evalmvx.yaml` | EvalMVX (Yang et al.) — 25 objects, 20 views |
| `config/lucesmv.yaml` | LUCES-MV |
| `config/default.yaml` | Base defaults; reference for every available key |

## Adding your own dataset

1. Arrange the data as in the [input layout](input-layout.md).
2. Copy `config/martine.yaml`, set `paths.eval_root` / `paths.data_root`, list your
   `dataset.objects` and `dataset.methods`.
3. Set the thresholds in **your mesh's units** — re-read
   [the units trap](configuration.md#the-units-trap-sk3d) if your meshes are not in millimetres.
4. Run the [end-to-end flow](pipeline.md).
