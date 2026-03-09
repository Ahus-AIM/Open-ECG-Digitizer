#!/bin/bash
# Auto-retrain U-Net once 20K mask rendering is complete.
# Waits for rendering to finish, then starts U-Net training with ALL data:
#   - 5K original ACS masks  (data/acs_multilayout_masks/)
#   - 20K new ACS masks      (data/acs_multilayout_masks_20k/)
#   - ~3.7K HuggingFace      (parquet)
# Total: ~29K images+masks
#
# Usage: bash scripts/auto_retrain_unet.sh &

set -e
cd /volume/Open-ECG-Digitizer

RENDER_DIR="data/acs_multilayout_masks_20k"
TARGET_COUNT=19000  # Wait until at least 19K rendered (allows for some errors)

echo "[$(date)] Waiting for rendering to finish..."
echo "  Target: $TARGET_COUNT images in $RENDER_DIR/images/"

while true; do
    COUNT=$(ls "$RENDER_DIR/images/" 2>/dev/null | wc -l)

    # Check if render process is still running
    RENDER_RUNNING=$(pgrep -f "render_acs_multilayout.*masks_20k" 2>/dev/null | wc -l)

    if [ "$COUNT" -ge "$TARGET_COUNT" ]; then
        echo "[$(date)] Rendering complete! $COUNT images found."
        break
    fi

    if [ "$RENDER_RUNNING" -eq 0 ] && [ "$COUNT" -gt 0 ]; then
        echo "[$(date)] Renderer stopped with $COUNT images."
        break
    fi

    echo "[$(date)] Rendering: $COUNT/$TARGET_COUNT images..."
    sleep 120
done

# Merge the 5K and 20K mask directories into a combined manifest
echo "[$(date)] Merging datasets..."
COMBINED_DIR="data/acs_multilayout_masks_combined"
mkdir -p "$COMBINED_DIR/images" "$COMBINED_DIR/masks"

# Symlink all images and masks from both sources
echo "  Linking 5K original masks..."
for f in data/acs_multilayout_masks/images/*.png; do
    bn=$(basename "$f")
    ln -sf "$(readlink -f $f)" "$COMBINED_DIR/images/$bn" 2>/dev/null || true
done
for f in data/acs_multilayout_masks/masks/*.png; do
    bn=$(basename "$f")
    ln -sf "$(readlink -f $f)" "$COMBINED_DIR/masks/$bn" 2>/dev/null || true
done

echo "  Linking 20K new masks..."
for f in "$RENDER_DIR/images/"*.png; do
    bn=$(basename "$f")
    ln -sf "$(readlink -f $f)" "$COMBINED_DIR/images/20k_$bn" 2>/dev/null || true
done
for f in "$RENDER_DIR/masks/"*.png; do
    bn=$(basename "$f")
    ln -sf "$(readlink -f $f)" "$COMBINED_DIR/masks/20k_$bn" 2>/dev/null || true
done

TOTAL_IMAGES=$(ls "$COMBINED_DIR/images/" | wc -l)
TOTAL_MASKS=$(ls "$COMBINED_DIR/masks/" | wc -l)
echo "  Combined: $TOTAL_IMAGES images, $TOTAL_MASKS masks"

# Start U-Net training on GPU 2
echo "[$(date)] Starting U-Net training on cuda:2 with full dataset..."
PYTHONUNBUFFERED=1 python scripts/train_unet_multilayout.py \
    --device cuda:2 \
    --prerendered_dir "$COMBINED_DIR" \
    --hf_parquet_dir /media/data1/datasets/Huggingface_ECG_Digitize/parquet/data/ \
    --save_dir weights/unet_full_combined/ \
    --init_weights weights/unet_combined/best_weights.pt \
    --batch_size 4 \
    --epochs 40 \
    --lr 5e-5 \
    --crop_size 1024 \
    --patience 15 \
    --num_workers 4 \
    --amp \
    2>&1 | tee "logs/train_unet_full_combined_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] U-Net training complete!"
