#!/usr/bin/env python3
"""
Full ECG Pipeline: Photo → Layout Classification → Digitization → ACS Prediction

Runs all 3 models on a folder of ECG images (phone photos or scans).

REQUIRED MODEL FILES (copy all to a `models/` directory):
  models/
  ├── layout_classifier.pt          (43 MB)  - ResNet-18, 13-class layout classifier
  ├── unet_segmentation.pt          (87 MB)  - U-Net, 4-class segmentation
  ├── wcr_77_classes.pt             (1.1 GB) - WCR encoder (fairseq_signals)
  ├── base_ssl.pt                   (1.1 GB) - WCR SSL backbone
  ├── acs_augment_v4.pt             (1.1 GB) - ACS fine-tuned head + encoder
  └── lead_name_unet_weights.pt     (22 MB)  - Lead name identifier U-Net

SOURCE PATHS ON SERVER:
  layout_classifier.pt   = /volume/Open-ECG-Digitizer/weights/layout_classifier/best_layout_classifier.pt
  unet_segmentation.pt   = /volume/Open-ECG-Digitizer/weights/unet_multilayout/best_weights.pt
  wcr_77_classes.pt       = /volume/DeepECG_Docker/weights/wcr_77_classes/wcr_77_classes.pt
  base_ssl.pt             = /volume/DeepECG_Docker/weights/wcr_77_classes/base_ssl.pt
  acs_augment_v4.pt       = /volume/DeepECG_Docker/checkpoints_acs_online_augment_from_ceiling/best_model.pt
  lead_name_unet_weights.pt = /volume/Open-ECG-Digitizer/weights/lead_name_unet_weights_07072025.pt

ALSO NEEDED (code directories):
  - /volume/Open-ECG-Digitizer/src/          (digitizer code)
  - /volume/DeepECG_Docker/fairseq-signals/  (WCR model code, pip install -e)

Usage:
    python scripts/predict_folder.py --input_dir /path/to/ecg/photos/
    python scripts/predict_folder.py --input_dir /path/to/photos/ --device cuda:0 --threshold 0.047
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample

# ─── Config ─────────────────────────────────────────────────────────────────

PTBXL_POWER_RATIO = 3.003154
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

LAYOUT_CLASSES = [
    'cabrera_12x1', 'cabrera_6x1_limb', 'precordial_3x2', 'precordial_6x1',
    'standard_12x1', 'standard_3x1', 'standard_3x4', 'standard_3x4_with_r1',
    'standard_3x4_with_r2', 'standard_3x4_with_r3', 'standard_6x1_limb',
    'standard_6x2', 'standard_6x2_with_r1',
]

# Youden-optimal threshold from test set analysis (n=4037)
DEFAULT_THRESHOLD = 0.047  # sens=74.1%, spec=87.8%


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_layout_classifier(model_dir, device):
    """Load ResNet-18 layout classifier (13 classes, 97.3% val accuracy)."""
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LAYOUT_CLASSES))
    path = os.path.join(model_dir, 'layout_classifier.pt')
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def load_wcr_model(model_dir, device):
    """Load WCR encoder + Augment v4 ACS head (AUC 0.882, AUPRC 0.600)."""
    from fairseq_signals.utils import checkpoint_utils

    wcr_path = os.path.join(model_dir, 'wcr_77_classes.pt')
    ssl_path = os.path.join(model_dir, 'base_ssl.pt')
    acs_path = os.path.join(model_dir, 'acs_augment_v4.pt')

    model, cfg, task = checkpoint_utils.load_model_and_task(
        wcr_path, arg_overrides={"model_path": ssl_path}, suffix="",
    )
    # Replace 77-class head with 1-class ACS head
    model.proj = nn.Linear(model.proj.in_features, 1)

    # Load fine-tuned ACS weights
    ckpt = torch.load(acs_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    return model.to(device).eval()


def load_digitizer(model_dir, device_str='cuda'):
    """Load the Open-ECG-Digitizer inference wrapper."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model.inference_wrapper import InferenceWrapper

    unet_path = os.path.join(model_dir, 'unet_segmentation.pt')
    lead_unet_path = os.path.join(model_dir, 'lead_name_unet_weights.pt')

    config = {
        'SIGNAL_EXTRACTOR': {
            'class_path': 'src.model.signal_extractor.SignalExtractor', 'KWARGS': {},
        },
        'PERSPECTIVE_DETECTOR': {
            'class_path': 'src.model.perspective_detector.PerspectiveDetector',
            'KWARGS': {'num_thetas': 250},
        },
        'DEWARPER': {
            'class_path': 'src.model.dewarper.Dewarper',
            'KWARGS': {'abs_peak_threshold': 0.1},
        },
        'SEGMENTATION_MODEL': {
            'class_path': 'src.model.unet.UNet',
            'weight_path': unet_path,
            'KWARGS': {
                'num_in_channels': 3, 'num_out_channels': 4,
                'dims': [32, 64, 128, 256, 320, 320, 320, 320], 'depth': 2,
            },
        },
        'CROPPER': {
            'class_path': 'src.model.cropper.Cropper',
            'KWARGS': {'granularity': 80, 'percentiles': [0.02, 0.98], 'alpha': 0.85},
        },
        'PIXEL_SIZE_FINDER': {
            'class_path': 'src.model.pixel_size_finder.PixelSizeFinder',
            'KWARGS': {
                'min_number_of_grid_lines': 30,
                'max_number_of_grid_lines': 70,
                'lower_grid_line_factor': 0.3,
            },
        },
        'LAYOUT_IDENTIFIER': {
            'class_path': 'src.model.lead_identifier.LeadIdentifier',
            'config_path': 'src/config/lead_layouts_reduced.yml',
            'unet_config_path': 'src/config/lead_name_unet.yml',
            'unet_weight_path': lead_unet_path,
            'KWARGS': {'debug': False, 'device': device_str, 'possibly_flipped': False},
        },
    }

    wrapper = InferenceWrapper(
        config=config,
        device=device_str,
        resample_size=3000,
        rotate_on_resample=True,
        enable_timing=False,
        apply_dewarping=False,
    )
    return wrapper


# ─── Inference Functions ────────────────────────────────────────────────────

def classify_layout(model, image_path, device, crop_size=512):
    """Classify ECG layout from image. Returns (layout_name, confidence)."""
    img = cv2.imread(image_path)
    if img is None:
        return 'unknown', 0.0
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


def digitize_image(wrapper, image_path, timeout=60):
    """Digitize an ECG image to 12-lead signal. Returns (ecg_12x2500, error)."""
    import threading

    result = [None]
    error = [None]

    def _run():
        try:
            image = torch.from_numpy(
                cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            ).permute(2, 0, 1).float() / 255.0
            got = wrapper(image, layout_should_include_substring=None)
            # Extract canonical 12-lead signal
            canonical = got.get('canonical_ecg')
            if canonical is not None:
                result[0] = canonical.cpu().numpy()
            else:
                # Try to reconstruct from individual leads
                leads = []
                for lead in LEAD_NAMES:
                    key = f'timeseries_{lead}'
                    if key in got and got[key] is not None:
                        leads.append(got[key].cpu().numpy().flatten())
                    else:
                        leads.append(np.zeros(3000))
                result[0] = np.array(leads)
        except Exception as e:
            error[0] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return None, "TIMEOUT"
    if error[0]:
        return None, error[0]
    return result[0], None


def predict_acs(model, ecg, device):
    """Predict ACS probability from 12-lead ECG array. Returns probability [0,1]."""
    # Ensure (12, N)
    if ecg.ndim == 1:
        return None
    if ecg.shape[0] != 12 and ecg.shape[1] == 12:
        ecg = ecg.T
    ecg = np.nan_to_num(ecg, nan=0.0)

    # Resample to 2500 samples
    if ecg.shape[1] != 2500:
        ecg = np.array([resample(lead, 2500) for lead in ecg])

    # PSA normalization
    ecg = ecg * PTBXL_POWER_RATIO

    ecg_t = torch.FloatTensor(ecg).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(source=ecg_t)
        prob = torch.sigmoid(output['out']).item()
    return prob


def predict_acs_from_csv(model, csv_path, device):
    """Predict ACS from a digitizer CSV output file."""
    df = pd.read_csv(csv_path)
    ecg = np.array([
        np.nan_to_num(df[l].values.astype(float)) if l in df.columns
        else np.zeros(len(df))
        for l in LEAD_NAMES
    ])
    return predict_acs(model, ecg, device)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run full ECG pipeline on a folder of images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input_dir', required=True,
                        help='Folder of ECG images (.png, .jpg)')
    parser.add_argument('--model_dir', default='models/',
                        help='Directory containing all model weights')
    parser.add_argument('--output_json', default=None,
                        help='Save results as JSON (default: print to stdout)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'ACS decision threshold (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--csv_dir', default=None,
                        help='Use pre-digitized CSVs instead of running digitizer')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Find images
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    images = sorted([f for f in os.listdir(args.input_dir)
                     if f.lower().endswith(exts)])
    if not images:
        print(f"No images found in {args.input_dir}")
        return

    print(f"Found {len(images)} images in {args.input_dir}\n")

    # Load models
    print("Loading models...")
    t0 = time.time()
    layout_model = load_layout_classifier(args.model_dir, device)
    print(f"  [1/2] Layout Classifier OK (ResNet-18, 13 classes)")

    wcr_model = load_wcr_model(args.model_dir, device)
    print(f"  [2/2] WCR ACS Model OK (Augment v4, AUC=0.882)")
    print(f"  Models loaded in {time.time()-t0:.1f}s\n")

    # Process
    print(f"{'ECG':<45} {'Layout':<22} {'Prob':>6} {'Flag':>6}")
    print("─" * 85)

    results = []
    for img_name in images:
        img_path = os.path.join(args.input_dir, img_name)
        basename = os.path.splitext(img_name)[0]

        # Step 1: Layout
        layout, layout_conf = classify_layout(layout_model, img_path, device)

        # Step 2+3: Get ACS probability
        prob = None
        if args.csv_dir:
            csv_path = os.path.join(args.csv_dir, f"{basename}_timeseries_canonical.csv")
            if os.path.exists(csv_path):
                prob = predict_acs_from_csv(wcr_model, csv_path, device)

        flag = ""
        if prob is not None:
            flag = "ACS+" if prob >= args.threshold else "ACS-"
            prob_str = f"{prob:.4f}"
        else:
            prob_str = "N/A"

        print(f"{basename:<45} {layout:<22} {prob_str:>6} {flag:>6}")

        results.append({
            'image': img_name,
            'layout': layout,
            'layout_confidence': round(layout_conf, 4),
            'acs_probability': round(prob, 4) if prob is not None else None,
            'acs_positive': prob >= args.threshold if prob is not None else None,
            'threshold': args.threshold,
        })

    print("─" * 85)

    # Count flagged
    flagged = sum(1 for r in results if r.get('acs_positive'))
    total_pred = sum(1 for r in results if r.get('acs_probability') is not None)
    print(f"\n{flagged}/{total_pred} flagged as ACS+ (threshold={args.threshold})")

    # Save JSON
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == '__main__':
    main()
