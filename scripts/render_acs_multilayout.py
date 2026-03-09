#!/usr/bin/env python3
"""
Render ACS NPYs in diverse layouts using multiprocessing.

Takes the ACS v6 dataset CSV, renders each NPY in a randomly selected layout,
saves PNGs + segmentation masks + layout labels CSV.

Output structure:
    output_dir/
        images/       # Rendered ECG images
        masks/        # Pixel-perfect segmentation masks
        manifest.csv  # npy_path, png_path, mask_path, layout, label

Usage:
    python scripts/render_acs_multilayout.py --workers 8
    python scripts/render_acs_multilayout.py --workers 8 --max_rows 1000  # quick test
"""

import argparse
import csv
import os
import random
import sys
import time
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '/volume/Open-ECG-Digitizer')
from scripts.multi_layout_renderer import (
    LAYOUTS, LEAD_NAMES, load_ecg_npy, render_ecg_with_mask,
)

# Layouts weighted by real-world frequency
# 3x4+R1 is most common, then 3x4+R3, 6x2, 12x1, etc.
LAYOUT_WEIGHTS = {
    'standard_3x4_with_r1': 30,
    'standard_3x4_with_r3': 15,
    'standard_3x4': 10,
    'standard_3x4_with_r2': 8,
    'standard_6x2': 10,
    'standard_6x2_with_r1': 5,
    'standard_12x1': 8,
    'cabrera_12x1': 4,
    'precordial_6x1': 3,
    'standard_6x1_limb': 2,
    'cabrera_6x1_limb': 2,
    'standard_3x1': 1,
    'precordial_3x2': 2,
}
WEIGHTED_LAYOUTS = []
for name, weight in LAYOUT_WEIGHTS.items():
    WEIGHTED_LAYOUTS.extend([name] * weight)

# Random visual styles
GRID_COLORS = ['red', '#FF6B6B', '#CC0000', '#FF4444',
               '#00AA00', '#006600', '#4488FF', '#2266CC', '#FF8800']


def render_one(args):
    """Render a single NPY file. Designed for multiprocessing Pool."""
    idx, npy_path, label, output_dir, with_mask = args
    # Support tuple with layout_suffix for multi-layout mode
    if isinstance(args, (list, tuple)) and len(args) == 6:
        idx, npy_path, label, output_dir, with_mask, layout_suffix = args
    else:
        layout_suffix = None

    try:
        lead_dict = load_ecg_npy(npy_path)
    except Exception as e:
        return None, f"LOAD_ERROR: {e}"

    # Random layout (weighted)
    layout = random.choice(WEIGHTED_LAYOUTS)

    # Random visual style
    grid_color = random.choice(GRID_COLORS)
    grid_alpha = random.uniform(0.5, 1.0)
    linewidth = random.uniform(1.5, 4.0)
    show_labels = random.random() > 0.1

    try:
        basename = os.path.splitext(os.path.basename(npy_path))[0]
        if layout_suffix is not None:
            basename = f"{basename}_{layout_suffix}"

        if with_mask:
            from scripts.multi_layout_renderer import render_ecg_with_mask
            img, mask = render_ecg_with_mask(
                lead_dict, layout,
                amplitude_factor=4.88, width=2500, random_rhythm=True,
                grid_color=grid_color, grid_alpha=grid_alpha,
                linewidth=linewidth, show_labels=show_labels,
            )
            mask_path = os.path.join(output_dir, 'masks', f"{basename}.png")
            mask.save(mask_path)
        else:
            from scripts.multi_layout_renderer import render_ecg
            img = render_ecg(
                lead_dict, layout,
                amplitude_factor=4.88, width=2500, random_rhythm=True,
                grid_color=grid_color, grid_alpha=grid_alpha,
                linewidth=linewidth, show_labels=show_labels,
            )
            mask_path = ''

        img_path = os.path.join(output_dir, 'images', f"{basename}.png")
        img.save(img_path)

        return {
            'npy_path': npy_path,
            'png_path': img_path,
            'mask_path': mask_path,
            'layout': layout,
            'condition_severity_modified': label,
        }, None

    except Exception as e:
        return None, f"RENDER_ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description='Render ACS NPYs in diverse layouts')
    parser.add_argument('--csv', default='/media/data1/ravram/DeepECG_Datasets/DEEPECG_ACS_FINAL_2017_2024_cath_ecg_merged_v6.csv')
    parser.add_argument('--output', '-o', default='/volume/Open-ECG-Digitizer/data/acs_multilayout/')
    parser.add_argument('--workers', '-w', type=int, default=8)
    parser.add_argument('--max_rows', type=int, default=None, help='Limit rows for testing')
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--layouts_per_signal', type=int, default=1,
                        help='Render each signal in N different random layouts')
    parser.add_argument('--with_masks', action='store_true',
                        help='Also generate segmentation masks (2x slower)')
    args = parser.parse_args()

    import pandas as pd
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False,
                     usecols=['npy_path', 'condition_severity_modified'])
    if args.max_rows:
        df = df.head(args.max_rows)
    print(f"Loaded {len(df)} rows ({(df.condition_severity_modified >= 2).sum()} positive)")

    # Create output dirs
    os.makedirs(os.path.join(args.output, 'images'), exist_ok=True)
    if args.with_masks:
        os.makedirs(os.path.join(args.output, 'masks'), exist_ok=True)

    # Prepare work items
    if args.layouts_per_signal > 1:
        # Each signal rendered in N different layouts
        work = []
        for i, row in df.iterrows():
            for k in range(args.layouts_per_signal):
                work.append((i, row.npy_path, row.condition_severity_modified,
                            args.output, args.with_masks, f"v{k}"))
        print(f"Multi-layout mode: {args.layouts_per_signal} layouts/signal → "
              f"{len(work)} total renders")
    else:
        work = [
            (i, row.npy_path, row.condition_severity_modified, args.output, args.with_masks)
            for i, row in df.iterrows()
        ]

    print(f"Rendering {len(work)} images with {args.workers} workers...")
    t0 = time.time()

    results = []
    errors = 0

    with Pool(args.workers) as pool:
        for i, (result, error) in enumerate(pool.imap_unordered(render_one, work,
                                                                 chunksize=args.chunk_size)):
            if error:
                errors += 1
                if errors <= 20:
                    print(f"  ERROR [{i}]: {error}")
            else:
                results.append(result)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(work) - i - 1) / rate / 60
                print(f"  [{i+1}/{len(work)}] {rate:.1f} img/s | "
                      f"ETA: {eta:.0f} min | Errors: {errors}")

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)}/{len(work)} images in {elapsed/60:.1f} min "
          f"({len(results)/elapsed:.1f} img/s, {errors} errors)")

    # Save manifest
    manifest_path = os.path.join(args.output, 'manifest.csv')
    keys = ['npy_path', 'png_path', 'mask_path', 'layout', 'condition_severity_modified']
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"Manifest: {manifest_path}")

    # Layout distribution
    from collections import Counter
    layout_counts = Counter(r['layout'] for r in results)
    print("\nLayout distribution:")
    for layout, count in sorted(layout_counts.items(), key=lambda x: -x[1]):
        print(f"  {layout}: {count}")


if __name__ == '__main__':
    main()
