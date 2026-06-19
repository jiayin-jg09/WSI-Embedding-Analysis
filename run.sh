#!/bin/bash
# =============================================================================
# WSI Embedding Analysis Pipeline - Launcher (Linux/Mac)
# =============================================================================
# Usage: bash run.sh [--cancer-type CHOL] [--n-folds 5]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists; create if not
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "No virtual environment found. Create one? (y/n)"
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        echo "Creating virtual environment..."
        python3 -m venv "$SCRIPT_DIR/venv"
        source "$SCRIPT_DIR/venv/bin/activate"
        echo "Installing dependencies..."
        pip install -r "$SCRIPT_DIR/requirements.txt"
    else
        echo "Proceeding without virtual environment..."
    fi
else
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "Activated virtual environment: $SCRIPT_DIR/venv"
fi

# Check that data exists
if [ ! -f "$SCRIPT_DIR/CLINICAL_FULL.parquet" ]; then
    echo "ERROR: CLINICAL_FULL.parquet not found in package directory."
    echo "Run 'bash setup_data.sh' first, or place the file manually."
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/embeddings" ] || [ -z "$(ls -A "$SCRIPT_DIR/embeddings" 2>/dev/null)" ]; then
    echo "WARNING: No embedding files found in embeddings/"
    echo "The script will run in demo mode (no actual analysis)."
fi

echo ""
echo "=============================================="
echo "Running WSI Embedding Analysis Pipeline"
echo "=============================================="
echo ""

python3 "$SCRIPT_DIR/wsi_embedding_analysis.py" \
    --embeddings-dir "$SCRIPT_DIR/embeddings" \
    --clinical-data "$SCRIPT_DIR/CLINICAL_FULL.parquet" \
    --output-dir "$SCRIPT_DIR/results" \
    --cancer-type CHOL \
    "$@"

echo ""
echo "Done! Results saved to: $SCRIPT_DIR/results/"
