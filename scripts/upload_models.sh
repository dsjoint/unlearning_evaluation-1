#!/bin/bash
# Helper script to upload models to HuggingFace Hub

set -e

echo "=== HuggingFace Model Upload Helper ==="
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN not set. Please provide your HuggingFace token."
    echo ""
    echo "To get a token:"
    echo "  1. Go to https://huggingface.co/settings/tokens"
    echo "  2. Create a new token with 'Write' permissions"
    echo "  3. Copy the token"
    echo ""
    echo "Then run:"
    echo "  export HF_TOKEN=your_token_here"
    echo "  bash upload_models.sh"
    echo ""
    echo "Or login interactively:"
    echo "  hf auth login"
    exit 1
fi

# Login with token
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
hf auth login --token "$HF_TOKEN" --add-to-git-credential

echo ""
echo "✅ Logged in to HuggingFace"
echo ""

# List available runs
echo "Available runs:"
python scripts/upload_models_to_hf.py --list

echo ""
read -p "Enter run name to upload (e.g., 2025-12-28_05-13-18): " RUN_NAME
read -p "Enter your HuggingFace username: " HF_USERNAME
read -p "Make repository private? (y/n): " PRIVATE

if [ "$PRIVATE" = "y" ] || [ "$PRIVATE" = "Y" ]; then
    PRIVATE_FLAG="--private"
else
    PRIVATE_FLAG=""
fi

echo ""
echo "Starting upload of $RUN_NAME..."
echo "This may take a while (40-50 GB per run)..."
echo ""

python scripts/upload_models_to_hf.py --run-name "$RUN_NAME" --hf-username "$HF_USERNAME" $PRIVATE_FLAG

echo ""
echo "✅ Upload complete!"

