#!/bin/bash
# =============================================================================
# Setup Data for WSI Embedding Analysis Pipeline
# =============================================================================
# This script copies the required data files into the portable package directory.
# Run this BEFORE transferring the package to an external drive.
#
# Usage: bash setup_data.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMBEDDINGS_SRC="/mnt/samsung990pro"
CLINICAL_SRC="/home/chrishartley/Projects/D Drive 9-5-2025/Public datasets/TCGA/TCGA_MODULES_FULL/CLINICAL_FULL.parquet"
EMBEDDINGS_DST="$SCRIPT_DIR/embeddings"

echo "=============================================="
echo "WSI Embedding Analysis - Data Setup"
echo "=============================================="
echo ""

# --- Step 1: Copy clinical data ---
echo "[1/3] Copying clinical data..."
if [ -f "$CLINICAL_SRC" ]; then
    cp "$CLINICAL_SRC" "$SCRIPT_DIR/CLINICAL_FULL.parquet"
    CLINICAL_SIZE=$(du -h "$SCRIPT_DIR/CLINICAL_FULL.parquet" | cut -f1)
    echo "  -> CLINICAL_FULL.parquet ($CLINICAL_SIZE)"
else
    echo "  ERROR: Clinical data not found at:"
    echo "    $CLINICAL_SRC"
    echo "  Please update CLINICAL_SRC in this script."
    exit 1
fi

# --- Step 2: Copy CHOL H5 embedding files ---
echo ""
echo "[2/3] Copying TCGA-CHOL embedding files..."
mkdir -p "$EMBEDDINGS_DST"

if [ ! -d "$EMBEDDINGS_SRC" ]; then
    echo "  ERROR: Embeddings source not found at:"
    echo "    $EMBEDDINGS_SRC"
    echo "  Please update EMBEDDINGS_SRC in this script."
    exit 1
fi

# Count available CHOL files (TCGA-W5 prefix is CHOL, but also match any TCGA H5)
# Copy all H5 files - the script filters by cancer type using clinical data
FILE_COUNT=0
for f in "$EMBEDDINGS_SRC"/*.h5 "$EMBEDDINGS_SRC"/*.hdf5; do
    if [ -f "$f" ]; then
        cp "$f" "$EMBEDDINGS_DST/"
        FILE_COUNT=$((FILE_COUNT + 1))
    fi
done

# Also check subdirectories (one level deep)
for d in "$EMBEDDINGS_SRC"/*/; do
    if [ -d "$d" ]; then
        for f in "$d"*.h5 "$d"*.hdf5; do
            if [ -f "$f" ]; then
                cp "$f" "$EMBEDDINGS_DST/"
                FILE_COUNT=$((FILE_COUNT + 1))
            fi
        done
    fi
done

if [ $FILE_COUNT -eq 0 ]; then
    echo "  WARNING: No H5/HDF5 files found in $EMBEDDINGS_SRC"
    echo "  You may need to copy embedding files manually into:"
    echo "    $EMBEDDINGS_DST/"
else
    EMBEDDINGS_SIZE=$(du -sh "$EMBEDDINGS_DST" | cut -f1)
    echo "  -> Copied $FILE_COUNT embedding files ($EMBEDDINGS_SIZE total)"
fi

# --- Step 3: Summary ---
echo ""
echo "[3/3] Summary"
echo "=============================================="
echo "  Package directory: $SCRIPT_DIR"
echo "  Clinical data:     CLINICAL_FULL.parquet"
echo "  Embeddings:        embeddings/ ($FILE_COUNT files)"
echo ""

# Show total package size
TOTAL_SIZE=$(du -sh "$SCRIPT_DIR" | cut -f1)
echo "  Total package size: $TOTAL_SIZE"
echo ""
echo "Done! You can now copy this entire directory to an external drive."
echo ""
echo "To run the analysis:"
echo "  cd $SCRIPT_DIR"
echo "  pip install -r requirements.txt"
echo "  bash run.sh"
