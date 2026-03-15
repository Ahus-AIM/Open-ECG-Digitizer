#!/bin/bash
# Auto-pilot: wait for rendering to finish, then train layout classifier
set -e

RENDER_DIR="/volume/Open-ECG-Digitizer/data/acs_multilayout_v3"
MANIFEST="$RENDER_DIR/manifest.csv"
TARGET=130000

echo "=== Waiting for rendering to complete ==="
echo "Target: $TARGET images"

while true; do
    COUNT=$(ls "$RENDER_DIR/images/" 2>/dev/null | wc -l)

    # Check if manifest exists (written at end of rendering)
    if [ -f "$MANIFEST" ]; then
        MANIFEST_ROWS=$(wc -l < "$MANIFEST")
        echo "[$(date '+%H:%M')] Manifest found with $MANIFEST_ROWS rows. Rendering complete!"
        break
    fi

    echo "[$(date '+%H:%M')] $COUNT / $TARGET images rendered..."
    sleep 300  # check every 5 min
done

echo ""
echo "=== Starting layout classifier training ==="
echo "Manifest: $MANIFEST"

cd /volume/Open-ECG-Digitizer

# Train on v3 data only (130K balanced images)
python scripts/train_layout_classifier.py \
    --manifest "$MANIFEST" \
    --save_dir weights/layout_classifier_v3/ \
    --device cuda:1 \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-3 \
    --crop_size 512 \
    --num_workers 4 \
    --patience 8

echo ""
echo "=== Training complete ==="
echo "Weights saved to: weights/layout_classifier_v3/"
