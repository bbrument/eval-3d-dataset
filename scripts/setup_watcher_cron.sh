#!/bin/bash
# Setup cron job for the evaluation pipeline watcher
# Usage: ./setup_watcher_cron.sh <config_file> [interval_minutes]

set -e

CONFIG_FILE="${1:-config/diligentmv.yaml}"
INTERVAL="${2:-15}"

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$PROJECT_DIR/$CONFIG_FILE"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Extract dataset name from config for log file
DATASET_NAME=$(basename "$CONFIG_FILE" .yaml)
LOG_DIR="/projects/m25115/eval_3d_datasets/$DATASET_NAME"
LOG_FILE="$LOG_DIR/eval_watcher.log"

# Create log directory if needed
mkdir -p "$LOG_DIR"

# Create the watcher script
WATCHER_SCRIPT="$PROJECT_DIR/scripts/run_watcher_${DATASET_NAME}.sh"
cat > "$WATCHER_SCRIPT" << EOF
#!/bin/bash
# Auto-generated watcher script for $DATASET_NAME
# Run by cron every $INTERVAL minutes

# Activate virtual environment
source "$PROJECT_DIR/venv/bin/activate"

# Run watcher
cd "$PROJECT_DIR"
eval-pipeline -c "$CONFIG_PATH" watch >> "$LOG_FILE" 2>&1

# Log completion
echo "[\$(date '+%Y-%m-%d %H:%M:%S')] Watcher scan completed" >> "$LOG_FILE"
EOF

chmod +x "$WATCHER_SCRIPT"

# Generate cron entry
CRON_ENTRY="*/$INTERVAL * * * * $WATCHER_SCRIPT"

echo "==========================================="
echo "Watcher Setup for: $DATASET_NAME"
echo "==========================================="
echo ""
echo "Config:    $CONFIG_PATH"
echo "Log file:  $LOG_FILE"
echo "Interval:  Every $INTERVAL minutes"
echo ""
echo "Watcher script created: $WATCHER_SCRIPT"
echo ""
echo "To add to crontab, run:"
echo ""
echo "  crontab -e"
echo ""
echo "Then add this line:"
echo ""
echo "  $CRON_ENTRY"
echo ""
echo "Or run this command to add automatically:"
echo ""
echo "  (crontab -l 2>/dev/null | grep -v 'run_watcher_${DATASET_NAME}'; echo '$CRON_ENTRY') | crontab -"
echo ""
echo "To test manually:"
echo ""
echo "  eval-pipeline -c $CONFIG_PATH watch --dry-run"
echo ""
echo "To view status:"
echo ""
echo "  eval-pipeline -c $CONFIG_PATH watch-status"
echo ""
echo "To view logs:"
echo ""
echo "  tail -f $LOG_FILE"
echo ""
