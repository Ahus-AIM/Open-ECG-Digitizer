#!/usr/bin/env python3
"""
Render balanced layout training data — equal samples per layout class.

Uses the matplotlib-based multi_layout_renderer (DeepECG_Preprocess style)
for high-quality ECG paper rendering with proper grid, colors, and labels.

Usage:
    python scripts/render_balanced_layouts.py --workers 12 --per_layout 10000
"""
import argparse
import csv
import os
import random
import sys
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '/volume/Open-ECG-Digitizer')
from scripts.multi_layout_renderer import (
    LAYOUTS, load_ecg_npy, render_ecg_random_style, render_ecg,
)

ALL_LAYOUTS = sorted(LAYOUTS.keys())  # 13 layouts


def render_one(args):
    idx, npy_path, label, layout, output_dir, figsize_scale, dpi = args
    try:
        lead_dict = load_ecg_npy(npy_path)
    except Exception as e:
        return None, f"LOAD_ERROR: {e}"

    try:
        basename = os.path.splitext(os.path.basename(npy_path))[0]
        basename = f"{basename}_{layout}"
        img_path = os.path.join(output_dir, 'images', f"{basename}.png")

        # Use random style for training diversity (varies grid color, linewidth, etc.)
        render_ecg_random_style(
            lead_dict, layout,
            amplitude_factor=4.88, width=2500,
            figsize_scale=figsize_scale, dpi=dpi,
            save_path=img_path,
        )

        return {
            'npy_path': npy_path,
            'png_path': img_path,
            'mask_path': '',
            'layout': layout,
            'condition_severity_modified': label,
        }, None
    except Exception as e:
        return None, f"RENDER_ERROR: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='/media/data1/ravram/DeepECG_Datasets/DEEPECG_ACS_FINAL_2017_2024_cath_ecg_merged_v6.csv')
    parser.add_argument('--output', '-o', default='/volume/Open-ECG-Digitizer/data/acs_multilayout_v3/')
    parser.add_argument('--workers', '-w', type=int, default=12)
    parser.add_argument('--per_layout', type=int, default=10000,
                        help='Target images per layout class')
    parser.add_argument('--chunk_size', type=int, default=5)
    parser.add_argument('--figsize_scale', type=float, default=0.4,
                        help='Figure size scale (0.4 = 40%% size, good balance of speed/quality)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='DPI for rendering')
    args = parser.parse_args()

    import pandas as pd
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False,
                     usecols=['npy_path', 'condition_severity_modified'])
    # Filter to existing NPYs
    df = df[df['npy_path'].notna()].reset_index(drop=True)
    print(f"Loaded {len(df)} rows")

    os.makedirs(os.path.join(args.output, 'images'), exist_ok=True)

    # Build work items: round-robin assign layouts to NPYs
    # Each layout gets exactly per_layout images
    npy_list = list(df.itertuples(index=False))
    random.seed(42)
    random.shuffle(npy_list)

    work = []
    for layout_idx, layout in enumerate(ALL_LAYOUTS):
        for i in range(args.per_layout):
            row = npy_list[i % len(npy_list)]
            work.append((len(work), row.npy_path, row.condition_severity_modified,
                        layout, args.output, args.figsize_scale, args.dpi))

    random.shuffle(work)  # shuffle for better multiprocessing distribution
    total = len(work)
    print(f"Rendering {total} images ({args.per_layout} × {len(ALL_LAYOUTS)} layouts) "
          f"with {args.workers} workers (figsize_scale={args.figsize_scale}, dpi={args.dpi})...")
    t0 = time.time()

    results = []
    errors = 0

    with Pool(args.workers) as pool:
        for i, (result, error) in enumerate(pool.imap_unordered(render_one, work,
                                                                 chunksize=args.chunk_size)):
            if error:
                errors += 1
                if errors <= 10:
                    print(f"  ERROR: {error}")
            else:
                results.append(result)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate / 60
                print(f"  [{i+1}/{total}] {rate:.1f} img/s | "
                      f"ETA: {eta:.0f} min | Errors: {errors}")

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)}/{total} in {elapsed/60:.1f} min "
          f"({len(results)/elapsed:.1f} img/s, {errors} errors)")

    manifest_path = os.path.join(args.output, 'manifest.csv')
    keys = ['npy_path', 'png_path', 'mask_path', 'layout', 'condition_severity_modified']
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"Manifest: {manifest_path}")

    from collections import Counter
    layout_counts = Counter(r['layout'] for r in results)
    print("\nLayout distribution:")
    for layout, count in sorted(layout_counts.items(), key=lambda x: -x[1]):
        print(f"  {layout}: {count}")


if __name__ == '__main__':
    main()
