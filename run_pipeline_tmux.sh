#!/bin/bash
# Run pipeline.py in a tmux session that you can detach/reattach
# Usage: ./run_pipeline_tmux.sh [session_name] [additional hydra overrides]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Session name (default: pipeline)
SESSION_NAME="${1:-pipeline}"
shift || true  # Remove first arg if provided, keep rest for hydra overrides

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting pipeline in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Detach:          Press Ctrl+B, then D"
echo "  Reattach:        tmux attach -t $SESSION_NAME"
echo "  List sessions:   tmux ls"
echo "  Kill session:    tmux kill-session -t $SESSION_NAME"
echo ""

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach -t "$SESSION_NAME"
else
    # Create new session and run pipeline
    tmux new-session -d -s "$SESSION_NAME" "cd '$SCRIPT_DIR' && python pipeline.py $*"
    echo "Session created. Attaching..."
    sleep 1
    tmux attach -t "$SESSION_NAME"
fi

