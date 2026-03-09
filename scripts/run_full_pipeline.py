#!/usr/bin/env python3
"""
Full 3-model pipeline: Layout Classifier → U-Net Digitizer → WCR ACS Classifier.

Step 1: Layout classifier (ResNet-18) identifies ECG layout
Step 2: Multilayout U-Net digitizes image → 12-lead signal
Step 3: Augment v4 WCR predicts acute coronary obstruction
"""

import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import cv2
import pandas as pd
from scipy.signal import resample

sys.path.insert(0, '/volume/Open-ECG-Digitizer')
sys.path.insert(0, '/volume/DeepECG_Docker')

PTBXL_POWER_RATIO = 3.003154

# ── Layout Classifier ──────────────────────────────────────────────────────

LAYOUT_CLASSES = [
    'cabrera_12x1', 'cabrera_6x1_limb', 'precordial_3x2', 'precordial_6x1',
    'standard_12x1', 'standard_3x1', 'standard_3x4', 'standard_3x4_with_r1',
    'standard_3x4_with_r2', 'standard_3x4_with_r3', 'standard_6x1_limb',
    'standard_6x2', 'standard_6x2_with_r1',
]

def load_layout_classifier(weight_path, device):
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LAYOUT_CLASSES))
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()

def classify_layout(model, image_path, device, crop_size=512):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = crop_size / min(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    h, w = img.shape[:2]
    y, x = (h - crop_size) // 2, (w - crop_size) // 2
    img = img[y:y+crop_size, x:x+crop_size]
    img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits = model(img_t.to(device))
        probs = torch.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()
    return LAYOUT_CLASSES[idx], probs[0, idx].item()


# ── WCR ACS Model ──────────────────────────────────────────────────────────

def load_wcr_model(device):
    from fairseq_signals.utils import checkpoint_utils
    wcr_path = '/volume/DeepECG_Docker/weights/wcr_77_classes/wcr_77_classes.pt'
    ssl_path = '/volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt'
    model, cfg, task = checkpoint_utils.load_model_and_task(
        wcr_path, arg_overrides={"model_path": ssl_path}, suffix="",
    )
    # Replace 77-class head with 1-class
    model.proj = nn.Linear(model.proj.in_features, 1)
    # Load augment v4 weights
    ckpt = torch.load(
        '/volume/DeepECG_Docker/checkpoints_acs_online_augment_from_ceiling/best_model.pt',
        map_location='cpu',
    )
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model.to(device).eval()

def predict_from_csv(model, csv_path, device):
    """Load digitized CSV, convert to 12-lead, run WCR."""
    df = pd.read_csv(csv_path)
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    ecg_leads = []
    for lead in lead_names:
        if lead in df.columns:
            vals = df[lead].values.astype(float)
            vals = np.nan_to_num(vals, nan=0.0)
            ecg_leads.append(vals)
        else:
            ecg_leads.append(np.zeros(len(df)))

    ecg = np.array(ecg_leads)  # (12, N)

    # Resample to 2500
    if ecg.shape[1] != 2500:
        ecg = np.array([resample(lead, 2500) for lead in ecg])

    # PSA normalization
    ecg = ecg * PTBXL_POWER_RATIO

    ecg_t = torch.FloatTensor(ecg).unsqueeze(0).to(device)
    with torch.no_grad():
        result = model(ecg_t)
        logit = result['out']
        prob = torch.sigmoid(logit).item()
    return prob


# ── Digitizer ──────────────────────────────────────────────────────────────

def run_digitizer(config_path):
    """Run digitizer as subprocess to avoid state conflicts."""
    import subprocess
    result = subprocess.run(
        ['python', '-c', f'''
import yaml, sys
sys.path.insert(0, "/volume/Open-ECG-Digitizer")
from yacs.config import CfgNode as CN
from src.digitize import main

with open("{config_path}") as f:
    cfg = CN(yaml.safe_load(f))
cfg.freeze()
main(cfg)
'''],
        capture_output=True, text=True, timeout=120, cwd='/volume/Open-ECG-Digitizer',
    )
    return result.returncode == 0, result.stderr[-500:] if result.stderr else ""


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import yaml
    import shutil

    device = torch.device('cuda:0')
    test_dir = '/volume/Open-ECG-Digitizer/sandbox/test_ecgs/'
    output_dir = '/volume/Open-ECG-Digitizer/sandbox/full_pipeline_output/'
    os.makedirs(output_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(test_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print("=" * 80)
    print("  FULL PIPELINE: Layout Classifier → Multilayout U-Net → Augment v4 WCR")
    print("=" * 80)

    # Load models
    print("\n[1/2] Loading Layout Classifier (ResNet-18, 97.3% val acc)...")
    layout_model = load_layout_classifier(
        'weights/layout_classifier/best_layout_classifier.pt', device)
    print("  OK")

    print("[2/2] Loading WCR + Augment v4 (AUC 0.882, AUPRC 0.600)...")
    wcr = load_wcr_model(device)
    print("  OK")

    # Step 1: Classify all layouts
    print(f"\n{'─'*80}")
    print("STEP 1: Layout Classification")
    print(f"{'─'*80}")
    layouts = {}
    for img_name in images:
        layout, conf = classify_layout(layout_model, os.path.join(test_dir, img_name), device)
        layouts[img_name] = (layout, conf)
        print(f"  {img_name:<45} → {layout:<25} ({conf:.1%})")

    # Step 2: Digitize with multilayout U-Net
    print(f"\n{'─'*80}")
    print("STEP 2: Digitization (Multilayout U-Net)")
    print(f"{'─'*80}")

    dig_output = os.path.join(output_dir, 'digitized')
    config = {
        'MODEL': {
            'class_path': 'src.model.inference_wrapper.InferenceWrapper',
            'KWARGS': {
                'config': {
                    'SIGNAL_EXTRACTOR': {'class_path': 'src.model.signal_extractor.SignalExtractor', 'KWARGS': {}},
                    'PERSPECTIVE_DETECTOR': {'class_path': 'src.model.perspective_detector.PerspectiveDetector', 'KWARGS': {'num_thetas': 250}},
                    'DEWARPER': {'class_path': 'src.model.dewarper.Dewarper', 'KWARGS': {'abs_peak_threshold': 0.1}},
                    'SEGMENTATION_MODEL': {
                        'class_path': 'src.model.unet.UNet',
                        'weight_path': './weights/unet_multilayout/best_weights.pt',
                        'KWARGS': {'num_in_channels': 3, 'num_out_channels': 4, 'dims': [32, 64, 128, 256, 320, 320, 320, 320], 'depth': 2},
                    },
                    'CROPPER': {'class_path': 'src.model.cropper.Cropper', 'KWARGS': {'granularity': 80, 'percentiles': [0.02, 0.98], 'alpha': 0.85}},
                    'PIXEL_SIZE_FINDER': {'class_path': 'src.model.pixel_size_finder.PixelSizeFinder', 'KWARGS': {'min_number_of_grid_lines': 30, 'max_number_of_grid_lines': 70, 'lower_grid_line_factor': 0.3}},
                    'LAYOUT_IDENTIFIER': {
                        'class_path': 'src.model.lead_identifier.LeadIdentifier',
                        'config_path': 'src/config/lead_layouts_reduced.yml',
                        'unet_config_path': 'src/config/lead_name_unet.yml',
                        'unet_weight_path': './weights/lead_name_unet_weights_07072025.pt',
                        'KWARGS': {'debug': False, 'device': 'cuda', 'possibly_flipped': False},
                    },
                },
                'device': 'cuda',
                'resample_size': 3000,
                'rotate_on_resample': True,
                'enable_timing': False,
                'apply_dewarping': False,
            },
        },
        'DATA': {
            'images_path': test_dir,
            'image_extensions': ['.png', '.jpg', '.jpeg', '.JPG'],
            'output_path': dig_output,
            'save_mode': 'all',
            'layout_should_include_substring': None,
        },
    }

    config_path = os.path.join(output_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"  Running digitizer on {len(images)} images...")
    t0 = time.time()
    success, err = run_digitizer(config_path)
    t_dig = time.time() - t0
    print(f"  {'OK' if success else 'ERRORS'} ({t_dig:.0f}s)")
    if not success:
        print(f"  Error: {err[-200:]}")

    # Step 3: WCR Prediction
    print(f"\n{'─'*80}")
    print("STEP 3: ACS Prediction (Augment v4 WCR)")
    print(f"{'─'*80}")

    print(f"\n{'ECG':<45} {'Layout':<25} {'Conf':>5}  {'ACS Prob':>8}  {'Label'}")
    print("─" * 100)

    for img_name in images:
        basename = os.path.splitext(img_name)[0]
        layout, conf = layouts[img_name]

        # Label
        if 'STEMI' in basename and 'NO' not in basename:
            label = 'STEMI (phone)'
        elif 'NO_STEMI' in basename:
            label = 'Normal (phone)'
        else:
            label = 'MHI scan'

        # Find CSV output
        csv_path = os.path.join(dig_output, f"{basename}_timeseries_canonical.csv")
        if os.path.exists(csv_path):
            try:
                prob = predict_from_csv(wcr, csv_path, device)
                prob_str = f"{prob:.4f}"
            except Exception as e:
                prob = None
                prob_str = f"ERR: {str(e)[:30]}"
        else:
            prob = None
            prob_str = "NO CSV"

        print(f"{basename:<45} {layout:<25} {conf:>5.1%}  {prob_str:>8}  {label}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Pipeline: ResNet-18 Layout (97.3%) → Multilayout U-Net → Augment v4 WCR (0.882)")
    print(f"Models used:")
    print(f"  1. Layout:  weights/layout_classifier/best_layout_classifier.pt")
    print(f"  2. U-Net:   weights/unet_multilayout/best_weights.pt")
    print(f"  3. WCR:     checkpoints_acs_online_augment_from_ceiling/best_model.pt")


if __name__ == '__main__':
    main()
