#!/bin/bash
# Copy all pipeline model files to a single directory for deployment.
#
# Usage:
#   bash scripts/copy_models.sh /path/to/output/models/
#
# This copies ~3.5 GB total (6 model files).

DEST="${1:-models/}"
mkdir -p "$DEST"

echo "Copying pipeline models to $DEST ..."

# 1. Layout Classifier (ResNet-18, 43 MB, 97.3% val accuracy)
cp -v /volume/Open-ECG-Digitizer/weights/layout_classifier/best_layout_classifier.pt \
      "$DEST/layout_classifier.pt"

# 2. U-Net Segmentation (87 MB, val loss 0.056)
cp -v /volume/Open-ECG-Digitizer/weights/unet_multilayout/best_weights.pt \
      "$DEST/unet_segmentation.pt"

# 3. WCR Encoder (1.1 GB)
cp -v /volume/DeepECG_Docker/weights/wcr_77_classes/wcr_77_classes.pt \
      "$DEST/wcr_77_classes.pt"

# 4. WCR SSL Backbone (1.1 GB)
cp -v /volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt \
      "$DEST/base_ssl.pt"

# 5. ACS Augment v4 Fine-tuned Weights (1.1 GB, AUC 0.882)
cp -v /volume/DeepECG_Docker/checkpoints_acs_online_augment_from_ceiling/best_model.pt \
      "$DEST/acs_augment_v4.pt"

# 6. Lead Name Identifier U-Net (22 MB)
cp -v /volume/Open-ECG-Digitizer/weights/lead_name_unet_weights_07072025.pt \
      "$DEST/lead_name_unet_weights.pt"

echo ""
echo "Done! Total:"
du -sh "$DEST"
echo ""
echo "Files:"
ls -lh "$DEST"
