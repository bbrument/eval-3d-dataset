# Running on a cluster

[← back to README](../README.md)

Set `execution.mode: "slurm"` and fill `execution.slurm.account` / `partition`. Job templates in
`slurm/templates/` (`cleanup.sh.j2`, `eval.sh.j2`) are fully parameterised — nothing
site-specific is baked into `src/`.

Machine-specific runtime setup (for example a headless OSMesa library and the C++ runtime it needs
on `LD_LIBRARY_PATH`, for `visualize`) goes in a shell script named by
`execution.slurm.setup_script`; every watcher job sources it first. Keep that script out of git:
copy `config/setup_env.example.sh` to `config/setup_env.local.sh` (git-ignored) and point your
`*.local.yaml` to it.

Dispatch the whole flow with `run --slurm`, or submit individual stages. `watch` polls for new
meshes and submits jobs as they appear; `watch-status` reports on them.

> Helper scripts under `scripts/` were written for one specific machine and still contain
> absolute paths. They are convenience wrappers, not part of the pipeline — `src/` contains no
> absolute path.
