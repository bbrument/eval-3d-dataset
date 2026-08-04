# Running on a cluster

[← back to README](../README.md)

Set `execution.mode: "slurm"` and fill `execution.slurm.account` / `partition`. Job templates in
`slurm/templates/` (`cleanup.sh.j2`, `eval.sh.j2`) are fully parameterised — nothing
site-specific is baked into `src/`.

Dispatch the whole flow with `run --slurm`, or submit individual stages. `watch` polls for new
meshes and submits jobs as they appear; `watch-status` reports on them.

> Helper scripts under `scripts/` were written for one specific machine and still contain
> absolute paths. They are convenience wrappers, not part of the pipeline — `src/` contains no
> absolute path.
