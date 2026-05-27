#!/bin/bash
# Auto-generated watcher script for dlmv
# Run by cron every 5 minutes

# Activate virtual environment
source "/home/babrument/dev/eval_dataset/eval_pipeline/venv/bin/activate"

# Run watcher
cd "/home/babrument/dev/eval_dataset/eval_pipeline"
eval-pipeline -c "/home/babrument/dev/eval_dataset/eval_pipeline/config/dlmv.yaml" watch >> "/projects/m25115/eval_3d_datasets/dlmv/eval_watcher.log" 2>&1

# Log completion
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher scan completed" >> "/projects/m25115/eval_3d_datasets/dlmv/eval_watcher.log"
