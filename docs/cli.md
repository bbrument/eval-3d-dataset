# Commands

Every `eval-pipeline` subcommand in one table; the stage-by-stage flow is in [pipeline.md](pipeline.md).

[← back to README](../README.md)

Every command takes `-c/--config` (required) and most take `-o/--object`, `-m/--method` and `-f/--force`.
`uv run eval-pipeline -c <config> <command> --help` lists the options of one command.

| Command | What it does |
| :-- | :-- |
| `preprocess-gt` | Clean the GT mesh, sample its point cloud, compute per-point attributes (not needed for Martine: `gt/` ships them) |
| `cleanup` | Carve reconstructed vertices that project outside every silhouette |
| `evaluate` | Distances, Chamfer, F-score → `eval_results/metrics.json` |
| `curves` | Dense precision / recall / F-score curves from the saved distances |
| `recompute` | Rebuild `metrics.json` and curves from saved distances (new `max_dist`, new mask) |
| `run` | `cleanup` + `evaluate` over the config's objects and methods (or `-o`, `-m`), locally or on SLURM (`--slurm`) |
| `apply-masks` | Per-zone distances from the GT attribute and challenge masks |
| `preprocess-challenges` | Build challenge masks from painted meshes or per-view 2D masks |
| `proxy-curvature` | GT curvature (Voronoi Covariance Measure) on a proxy mesh, tile by tile |
| `aggregate` | Collect per-cell metrics into tables across objects and methods |
| `figures` | Build a paper's figures and tables from a YAML specification |
| `latex` | LaTeX Chamfer table from the aggregated metrics |
| `visualize` | Render per-point error colourings (needs the `visualization` extra) |
| `watch`, `watch-status` | Poll for new meshes and submit evaluation jobs to SLURM; report on them |
