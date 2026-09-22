#!/bin/bash
# Launch all GT viz + method viz with --nice=100
# Usage: bash launch_all_viz.sh

set -euo pipefail

# Clear Python bytecode cache to ensure latest code is used
find /home/babrument/dev/eval_dataset/eval_pipeline/src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

CONFIG=/home/babrument/dev/eval_dataset/eval_pipeline/config/martine_watcher.local.yaml
VENV=/home/babrument/dev/eval_dataset/eval_pipeline/venv/bin/activate
MESA=/apps/spack/spack-softwares/linux-rocky9-zen3/gcc-13.1.0/mesa-23.3.6-topby2nfuloy3ucydjszrde2j4mmu57w
WRAP="export LD_LIBRARY_PATH=$MESA/lib:\${LD_LIBRARY_PATH:-} && export PYOPENGL_PLATFORM=osmesa && source $VENV"
EVAL_ROOT="/home/babrument/dev/alicevision/dataset_temp/eval"
NICE="--nice=100"

total_gtviz=0
total_viz=0

for obj_dir in $EVAL_ROOT/[0-9]*/; do
    obj=$(basename "$obj_dir")
    gt_dir="$obj_dir/Groundtruth"
    log_dir="$gt_dir/slurm_logs"
    mkdir -p "$log_dir"

    # Skip if no gt_pcd
    [ ! -f "$gt_dir/gt_pcd.npy" ] && { echo "SKIP $obj: no gt_pcd"; continue; }
    # Skip if no GT mesh
    gt_mesh=$(ls "$gt_dir"/gt_*.ply "$gt_dir"/*.ply 2>/dev/null | head -1)
    [ -z "$gt_mesh" ] && { echo "SKIP $obj: no GT mesh"; continue; }

    # Find a method for cameras
    method=$(find "$obj_dir" -maxdepth 2 -name "results_raw" -type d 2>/dev/null | head -1 | xargs -r dirname | xargs -r basename)
    if [ -z "$method" ]; then
        echo "SKIP $obj: no method for cameras"
        continue
    fi

    # Phase 1: GT viz (uniform + curvature + visibility, saves bbox.json)
    gtviz_jid=$(sbatch --parsable --kill-on-invalid-dep=yes $NICE \
        --account=m25115 --partition=mesonet \
        --cpus-per-task=4 --mem=128G --time=02:00:00 \
        --job-name="gtviz_${obj}" \
        --output="$log_dir/gtviz_%j.out" --error="$log_dir/gtviz_%j.err" \
        --wrap "$WRAP && eval-pipeline -c $CONFIG visualize -o $obj -m $method --metrics uniform,visibility,curvature -f")
    total_gtviz=$((total_gtviz+1))

    # Phase 2: method viz for all methods with eval done
    for method_dir in "$obj_dir"/*/; do
        m=$(basename "$method_dir")
        [[ "$m" == "Groundtruth" || "$m" =~ ^normals || "$m" =~ ^camera || "$m" =~ ^slurm || "$m" =~ ^debug ]] && continue
        [ ! -f "$method_dir/eval_results/metrics.json" ] && continue

        m_log="$method_dir/slurm_logs"
        mkdir -p "$m_log"

        sbatch --parsable --kill-on-invalid-dep=yes $NICE \
            --account=m25115 --partition=mesonet \
            --cpus-per-task=4 --mem=128G --time=01:00:00 \
            --dependency=afterok:$gtviz_jid \
            --job-name="viz_${obj}_${m}" \
            --output="$m_log/viz_%j.out" --error="$m_log/viz_%j.err" \
            --wrap "$WRAP && eval-pipeline -c $CONFIG visualize -o $obj -m $m -f" > /dev/null

        total_viz=$((total_viz+1))
    done

    echo "$obj: gtviz=$gtviz_jid → method viz submitted"
done

echo ""
echo "Total: $total_gtviz gtviz + $total_viz method viz = $((total_gtviz + total_viz)) jobs (nice=100)"
