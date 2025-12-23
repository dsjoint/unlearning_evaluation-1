#!/bin/bash
# Wrapper script for running pipeline.py with output logging
# Automatically places output logs in the logs/ directory

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for unique log filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Determine config name from args for log naming
CONFIG_NAME="default"
for arg in "$@"; do
    if [[ "$arg" == "--config-name="* ]]; then
        CONFIG_NAME="${arg#--config-name=}"
    fi
done

# Set log file path
LOG_FILE="logs/pipeline_${CONFIG_NAME}_${TIMESTAMP}.out"

echo "Starting pipeline..."
echo "Log file: $LOG_FILE"
echo "Arguments: $@"
echo "---"

# Run the pipeline and capture output
python pipeline.py "$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "---"
echo "Pipeline finished with exit code: $EXIT_CODE"
echo "Full log saved to: $LOG_FILE"

exit $EXIT_CODE

