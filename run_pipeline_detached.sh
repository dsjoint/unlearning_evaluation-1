#!/bin/bash
# Run pipeline.py in a detached mode that survives SSH disconnection
# Usage: ./run_pipeline_detached.sh [additional hydra overrides]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"

echo "Starting pipeline in detached mode..."
echo "Log file: $LOG_FILE"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To check if still running: ps aux | grep pipeline.py"
echo ""

# Run with nohup, redirecting both stdout and stderr to log file
# The pipeline uses conf/default.yaml by default
# To use a different config: ./run_pipeline_detached.sh --config-name=just_eval
# To override values: ./run_pipeline_detached.sh datasets=[YEARS] wandb_project_name="exp"
nohup python pipeline.py "$@" > "$LOG_FILE" 2>&1 &

# Get the PID
PID=$!

echo "Pipeline started with PID: $PID"
echo "Process will continue running even if you disconnect."
echo ""
echo "Useful commands:"
echo "  Monitor logs:     tail -f $LOG_FILE"
echo "  Check status:      ps -p $PID"
echo "  Kill process:      kill $PID"
echo "  View last 50 lines: tail -n 50 $LOG_FILE"

