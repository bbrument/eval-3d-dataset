# Machine-specific runtime setup, sourced at the start of every SLURM job when
# `execution.slurm.setup_script` points to it. Copy to config/setup_env.local.sh (git-ignored) and edit.
#
# Example: headless rendering for `visualize` with an OSMesa build that is not on the default library path.
# export LD_LIBRARY_PATH=/path/to/mesa/lib:${LD_LIBRARY_PATH:-}
# If that libOSMesa needs a newer libstdc++ than the system one, prepend its C++ runtime as well:
# export LD_LIBRARY_PATH=/path/to/gcc-runtime/lib:${LD_LIBRARY_PATH:-}
