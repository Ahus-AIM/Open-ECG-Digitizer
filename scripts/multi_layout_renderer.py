#!/usr/bin/env python3
"""
Multi-Layout ECG Renderer

Renders NPY ECG signals into PNG images using any of 13 standard ECG layouts
matching Open-ECG-Digitizer's lead_layouts_all.yml.

Two primary uses:
1. Layout classifier training data: render same NPY in multiple layouts
2. Segmentation U-Net training: diverse layouts + phone augmentations

Usage:
    # Render single NPY in a specific layout
    python multi_layout_renderer.py /path/to/ecg.npy --layout standard_3x4_with_r1

    # Render single NPY in ALL 13 layouts
    python multi_layout_renderer.py /path/to/ecg.npy --all-layouts -o /tmp/test_layouts/

    # Batch render folder of NPYs in all layouts (layout classifier data)
    python multi_layout_renderer.py /path/to/npy_folder/ --all-layouts -o /path/to/output/ --n-per-layout 500

    # Batch render with random rendering variations
    python multi_layout_renderer.py /path/to/npy_folder/ --all-layouts --random-style -o output/
"""

import argparse
import io
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image

# Standard 12-lead order (matches MHI NPY column order)
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Layout definitions matching Open-ECG-Digitizer's lead_layouts_all.yml
LAYOUTS = {
    'standard_3x4': {
        'grid_rows': 3, 'cols': 4,
        'leads': [
            ['I', 'aVR', 'V1', 'V4'],
            ['II', 'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
        ],
        'rhythm_leads': [],
    },
    'standard_3x4_with_r1': {
        'grid_rows': 3, 'cols': 4,
        'leads': [
            ['I', 'aVR', 'V1', 'V4'],
            ['II', 'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
        ],
        'rhythm_leads': ['II'],
    },
    'standard_3x4_with_r2': {
        'grid_rows': 3, 'cols': 4,
        'leads': [
            ['I', 'aVR', 'V1', 'V4'],
            ['II', 'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
        ],
        'rhythm_leads': ['II', 'V5'],
    },
    'standard_3x4_with_r3': {
        'grid_rows': 3, 'cols': 4,
        'leads': [
            ['I', 'aVR', 'V1', 'V4'],
            ['II', 'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
        ],
        'rhythm_leads': ['II', 'V1', 'V5'],
    },
    'standard_6x2': {
        'grid_rows': 6, 'cols': 2,
        'leads': [
            ['I', 'V1'],
            ['II', 'V2'],
            ['III', 'V3'],
            ['aVR', 'V4'],
            ['aVL', 'V5'],
            ['aVF', 'V6'],
        ],
        'rhythm_leads': [],
    },
    'standard_6x2_with_r1': {
        'grid_rows': 6, 'cols': 2,
        'leads': [
            ['I', 'V1'],
            ['II', 'V2'],
            ['III', 'V3'],
            ['aVR', 'V4'],
            ['aVL', 'V5'],
            ['aVF', 'V6'],
        ],
        'rhythm_leads': ['II'],
    },
    'standard_12x1': {
        'grid_rows': 12, 'cols': 1,
        'leads': [['I'], ['II'], ['III'], ['aVR'], ['aVL'], ['aVF'],
                  ['V1'], ['V2'], ['V3'], ['V4'], ['V5'], ['V6']],
        'rhythm_leads': [],
    },
    'cabrera_12x1': {
        'grid_rows': 12, 'cols': 1,
        'leads': [['aVL'], ['I'], ['-aVR'], ['II'], ['aVF'], ['III'],
                  ['V1'], ['V2'], ['V3'], ['V4'], ['V5'], ['V6']],
        'rhythm_leads': [],
    },
    'precordial_6x1': {
        'grid_rows': 6, 'cols': 1,
        'leads': [['V1'], ['V2'], ['V3'], ['V4'], ['V5'], ['V6']],
        'rhythm_leads': [],
    },
    'standard_6x1_limb': {
        'grid_rows': 6, 'cols': 1,
        'leads': [['I'], ['II'], ['III'], ['aVR'], ['aVL'], ['aVF']],
        'rhythm_leads': [],
    },
    'cabrera_6x1_limb': {
        'grid_rows': 6, 'cols': 1,
        'leads': [['aVL'], ['I'], ['-aVR'], ['II'], ['aVF'], ['III']],
        'rhythm_leads': [],
    },
    'standard_3x1': {
        'grid_rows': 3, 'cols': 1,
        'leads': [['I'], ['II'], ['III']],
        'rhythm_leads': [],
    },
    'precordial_3x2': {
        'grid_rows': 3, 'cols': 2,
        'leads': [['V1', 'V4'], ['V2', 'V5'], ['V3', 'V6']],
        'rhythm_leads': [],
    },
}


def load_ecg_npy(npy_path: str) -> dict:
    """Load NPY file and return lead dictionary.

    Handles shapes: (2500, 12), (2500, 12, 1), (12, 2500)
    """
    data = np.squeeze(np.load(npy_path))  # (2500, 12)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    if data.shape[0] == 12 and data.shape[1] != 12:
        data = data.T  # (12, N) → (N, 12)

    lead_dict = {name: data[:, i] for i, name in enumerate(LEAD_NAMES)}
    lead_dict['-aVR'] = -lead_dict['aVR']
    return lead_dict


def render_ecg(
    lead_dict: dict,
    layout_name: str = 'standard_3x4_with_r1',
    amplitude_factor: float = 4.88,
    width: int = 2500,
    grid_color: str = 'red',
    grid_alpha: float = 1.0,
    show_labels: bool = True,
    show_grid: bool = True,
    linewidth: float = 3.0,
    line_color: str = '#000000',
    title: str = '',
    random_rhythm: bool = False,
    label_fontsize: int = 28,
    label_color: str = 'black',
    background_color: str = 'white',
    mask_mode: bool = False,
    _resolved_rhythm: list = None,
    figsize_scale: float = 1.0,
    dpi: int = 100,
    save_path: str = None,
) -> Image.Image:
    """Render ECG leads into a PIL Image using the specified layout.

    Args:
        lead_dict: Dict mapping lead names ('I', 'II', ..., '-aVR') to 1D arrays
        layout_name: One of the 13 layout names in LAYOUTS
        amplitude_factor: Signal scaling (4.88 for raw MHI, 1000 for FFT-normalized)
        width: Output image width in pixels
        grid_color: ECG paper grid color
        grid_alpha: Grid opacity (0-1)
        show_labels: Show lead name labels
        show_grid: Show ECG paper grid
        linewidth: Signal trace line width
        line_color: Signal trace color
        title: Optional title text
        random_rhythm: Randomly pick leads for rhythm strips
        label_fontsize: Lead label font size
        label_color: Color for lead labels and calibration markers
        background_color: Figure background color
        mask_mode: If True, use mask colors (red=grid, green=text, blue=signal, black=bg)
        _resolved_rhythm: Pre-resolved rhythm leads (for consistency between img/mask)
        figsize_scale: Scale factor for figure size (0.5 = half size, faster rendering)
        dpi: DPI for rendering (lower = faster, 72 is good for training data)
        save_path: If set, save directly to this path and return None (avoids BytesIO overhead)

    Returns:
        PIL Image of rendered ECG (or None if save_path is set)
    """
    # Mask mode overrides colors for segmentation mask generation
    if mask_mode:
        background_color = 'black'
        grid_color = '#FF0000'
        line_color = '#0000FF'
        label_color = '#00FF00'
        grid_alpha = 1.0
        show_grid = True
        show_labels = True

    layout = LAYOUTS[layout_name]
    grid_rows = layout['grid_rows']
    cols = layout['cols']
    rhythm_cfg = layout['rhythm_leads']

    # Resolve rhythm leads
    if _resolved_rhythm is not None:
        rhythm_leads = _resolved_rhythm
    elif random_rhythm and rhythm_cfg:
        rhythm_leads = [random.choice(LEAD_NAMES) for _ in rhythm_cfg]
    else:
        rhythm_leads = list(rhythm_cfg)

    total_rows = grid_rows + len(rhythm_leads)
    n_samples = 2500
    spc = n_samples // cols

    # Calibration pulse: 5 zero + 50 high + 5 zero = 60 samples
    cal = [0] * 5 + [10] * 50 + [0] * 5
    cal_len = len(cal)

    # Y offsets — evenly spaced, centered at 0
    row_spacing = 35
    y_offsets = []
    for i in range(total_rows):
        y = (total_rows - 1 - i) * row_spacing - ((total_rows - 1) * row_spacing / 2)
        y_offsets.append(y)

    # Figure size: scale height with rows
    fig_w = 40 * figsize_scale
    fig_h = max(5 * total_rows, 10) * figsize_scale
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Grid
    if show_grid:
        ax.minorticks_on()
        ax.grid(ls='-', color=grid_color, linewidth=1.2, alpha=grid_alpha)
        ax.grid(which='minor', ls=':', color=grid_color, linewidth=1,
                alpha=grid_alpha * 0.8)

    # Axes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if mask_mode:
        ax.tick_params(colors=background_color)
        for spine in ax.spines.values():
            spine.set_color(background_color)

    all_ys = []

    # --- Plot grid rows ---
    for row_idx in range(grid_rows):
        y_off = y_offsets[row_idx]
        panel_y = [y_off + c for c in cal]  # Calibration pulse

        for col_idx in range(cols):
            lead_name = layout['leads'][row_idx][col_idx]
            signal = lead_dict.get(lead_name, np.zeros(n_samples))

            # First column starts after cal pulse; others at their time window
            start = cal_len if col_idx == 0 else col_idx * spc
            end = (col_idx + 1) * spc
            seg = signal[start:end]
            panel_y.extend([(s * amplitude_factor / 100) + y_off for s in seg])

        x = list(range(len(panel_y)))
        ax.plot(x, panel_y, linewidth=linewidth, color=line_color)
        all_ys.extend(panel_y)

        # Lead labels
        if show_labels:
            for col_idx in range(cols):
                lead_name = layout['leads'][row_idx][col_idx]
                lx = cal_len if col_idx == 0 else col_idx * spc
                ly = y_off + 5
                ax.vlines(lx, ly - 10, ly, linewidth=4, color=label_color)
                ax.text(lx + 5, ly, lead_name, fontsize=label_fontsize, color=label_color)

    # --- Plot rhythm leads (full-width strips at bottom) ---
    for r_idx, r_name in enumerate(rhythm_leads):
        y_off = y_offsets[grid_rows + r_idx]
        signal = lead_dict.get(r_name, np.zeros(n_samples))

        panel_y = [y_off + c for c in cal]
        panel_y.extend([(s * amplitude_factor / 100) + y_off for s in signal[cal_len:]])

        x = list(range(len(panel_y)))
        ax.plot(x, panel_y, linewidth=linewidth, color=line_color)
        all_ys.extend(panel_y)

        if show_labels:
            lx = cal_len
            ly = y_off + 5
            ax.vlines(lx, ly - 10, ly, linewidth=4, color=label_color)
            ax.text(lx + 5, ly, r_name, fontsize=label_fontsize, color=label_color)

    # Axis limits
    ax.set_xlim(-100, n_samples + 100)
    y_margin = 15
    ax.set_ylim(min(all_ys) - y_margin, max(all_ys) + y_margin)

    if title:
        ax.set_title(title, fontsize=30, color=label_color,
                     bbox=dict(facecolor=background_color, edgecolor=background_color,
                               boxstyle='round,pad=0.3'))

    plt.tight_layout()

    bg = background_color if background_color != 'white' else 'white'

    # Direct save to file (faster — avoids BytesIO + PIL reparse)
    if save_path is not None:
        plt.savefig(save_path, format='png', dpi=dpi, facecolor=bg, edgecolor='none')
        plt.close(fig)
        return None

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, facecolor=bg, edgecolor='none')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()

    # Resize to target width
    aspect = img.height / img.width
    new_h = int(width * aspect)
    resample = Image.NEAREST if mask_mode else Image.LANCZOS
    img = img.resize((width, new_h), resample)

    plt.close(fig)
    return img


def render_ecg_with_mask(
    lead_dict: dict,
    layout_name: str = 'standard_3x4_with_r1',
    amplitude_factor: float = 4.88,
    width: int = 2500,
    random_rhythm: bool = False,
    **image_kwargs,
) -> tuple:
    """Render ECG image AND segmentation mask with identical layout.

    Mask colors: Red=grid, Green=text, Blue=signal, Black=background.
    Matches Open-ECG-Digitizer's DiceFocalLoss rgb_to_one_hot() encoding.

    Returns:
        (image: PIL.Image, mask: PIL.Image) tuple
    """
    layout = LAYOUTS[layout_name]
    rhythm_cfg = layout['rhythm_leads']

    # Pre-resolve rhythm leads for consistency between image and mask
    if random_rhythm and rhythm_cfg:
        resolved = [random.choice(LEAD_NAMES) for _ in rhythm_cfg]
    else:
        resolved = list(rhythm_cfg)

    # Render image
    img = render_ecg(
        lead_dict, layout_name, amplitude_factor=amplitude_factor,
        width=width, _resolved_rhythm=resolved, **image_kwargs,
    )

    # Render mask with same layout but mask colors
    mask = render_ecg(
        lead_dict, layout_name, amplitude_factor=amplitude_factor,
        width=width, mask_mode=True, _resolved_rhythm=resolved,
    )

    return img, mask


def render_ecg_random_style(
    lead_dict: dict,
    layout_name: str = 'standard_3x4_with_r1',
    amplitude_factor: float = 4.88,
    width: int = 2500,
    figsize_scale: float = 1.0,
    dpi: int = 100,
    save_path: str = None,
) -> Image.Image:
    """Render ECG with randomized visual style for training data diversity.

    Randomly varies: grid color, grid opacity, line width, label presence,
    and rhythm lead selection.
    """
    grid_colors = ['red', '#FF6B6B', '#CC0000', '#FF4444',
                   '#00AA00', '#006600',  # green grid
                   '#4488FF', '#2266CC',  # blue grid
                   '#FF8800', '#CC6600']  # orange grid
    grid_color = random.choice(grid_colors)
    grid_alpha = random.uniform(0.5, 1.0)
    show_grid = random.random() > 0.05  # 95% have grid
    show_labels = random.random() > 0.1  # 90% have labels
    linewidth = random.uniform(1.5, 4.0)
    line_color = random.choice(['#000000', '#111111', '#222222', '#000044'])
    label_fontsize = random.randint(20, 36)

    # Slight amplitude variation (±15%)
    amp = amplitude_factor * random.uniform(0.85, 1.15)

    return render_ecg(
        lead_dict, layout_name, amplitude_factor=amp, width=width,
        grid_color=grid_color, grid_alpha=grid_alpha,
        show_labels=show_labels, show_grid=show_grid,
        linewidth=linewidth, line_color=line_color,
        random_rhythm=True, label_fontsize=label_fontsize,
        figsize_scale=figsize_scale, dpi=dpi, save_path=save_path,
    )


def batch_render(
    npy_dir: str,
    output_dir: str,
    layouts: list = None,
    n_per_layout: int = None,
    amplitude_factor: float = 4.88,
    width: int = 2500,
    random_style: bool = False,
):
    """Batch render NPY files in multiple layouts.

    Output structure:
        output_dir/
            standard_3x4/
                ecg_001.png
                ecg_002.png
            standard_3x4_with_r1/
                ecg_001.png
                ...

    Args:
        npy_dir: Directory containing .npy files
        output_dir: Root output directory
        layouts: List of layout names (default: all 13)
        n_per_layout: Max files per layout (default: all)
        amplitude_factor: Signal scaling
        width: Output image width
        random_style: Randomize visual style per image
    """
    if layouts is None:
        layouts = list(LAYOUTS.keys())

    os.makedirs(output_dir, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
    if n_per_layout:
        npy_files = npy_files[:n_per_layout]

    total = len(npy_files) * len(layouts)
    count = 0
    errors = 0

    print(f"Rendering {len(npy_files)} NPY files x {len(layouts)} layouts = {total} images")

    for layout_name in layouts:
        layout_dir = os.path.join(output_dir, layout_name)
        os.makedirs(layout_dir, exist_ok=True)

        for npy_file in npy_files:
            npy_path = os.path.join(npy_dir, npy_file)
            try:
                lead_dict = load_ecg_npy(npy_path)

                if random_style:
                    img = render_ecg_random_style(lead_dict, layout_name,
                                                  amplitude_factor, width)
                else:
                    img = render_ecg(lead_dict, layout_name, amplitude_factor,
                                    width, random_rhythm=True)

                out_name = npy_file.replace('.npy', '.png')
                img.save(os.path.join(layout_dir, out_name))
                count += 1

                if count % 50 == 0:
                    print(f"  [{count}/{total}] {layout_name}/{out_name}")
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  ERROR {npy_file}: {e}")

    print(f"\nDone: {count}/{total} images saved to {output_dir} ({errors} errors)")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Layout ECG Renderer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported layouts (13 total):
  standard_3x4          standard_3x4_with_r1   standard_3x4_with_r2
  standard_3x4_with_r3  standard_6x2           standard_6x2_with_r1
  standard_12x1         cabrera_12x1           precordial_6x1
  standard_6x1_limb     cabrera_6x1_limb       standard_3x1
  precordial_3x2
""")
    parser.add_argument('input', help='NPY file or folder of NPYs')
    parser.add_argument('--output', '-o', default='rendered_ecgs',
                        help='Output directory')
    parser.add_argument('--layout', '-l', default='standard_3x4_with_r1',
                        choices=list(LAYOUTS.keys()),
                        help='Layout (single file mode)')
    parser.add_argument('--all-layouts', action='store_true',
                        help='Render all 13 layouts')
    parser.add_argument('--layouts', nargs='+', choices=list(LAYOUTS.keys()),
                        help='Specific layouts to render')
    parser.add_argument('--n-per-layout', type=int, default=None,
                        help='Max files per layout (folder mode)')
    parser.add_argument('--width', type=int, default=2500,
                        help='Output image width in pixels')
    parser.add_argument('--amplitude', type=float, default=4.88,
                        help='Amplitude factor (4.88=raw MHI, 1000=FFT-normalized)')
    parser.add_argument('--random-style', action='store_true',
                        help='Randomize grid color, line width, etc. per image')
    parser.add_argument('--random-rhythm', action='store_true',
                        help='Randomize rhythm lead selection')
    args = parser.parse_args()

    if os.path.isfile(args.input):
        # Single file mode
        lead_dict = load_ecg_npy(args.input)

        if args.all_layouts:
            layouts = list(LAYOUTS.keys())
        elif args.layouts:
            layouts = args.layouts
        else:
            layouts = [args.layout]

        os.makedirs(args.output, exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.input))[0]

        for layout_name in layouts:
            if args.random_style:
                img = render_ecg_random_style(lead_dict, layout_name,
                                              args.amplitude, args.width)
            else:
                img = render_ecg(lead_dict, layout_name, args.amplitude,
                                 args.width, random_rhythm=args.random_rhythm)

            out_path = os.path.join(args.output, f'{basename}_{layout_name}.png')
            img.save(out_path)
            print(f"Saved: {out_path} ({img.size[0]}x{img.size[1]})")

    elif os.path.isdir(args.input):
        # Folder mode
        layouts = (list(LAYOUTS.keys()) if args.all_layouts
                   else (args.layouts or [args.layout]))
        batch_render(args.input, args.output, layouts, args.n_per_layout,
                     args.amplitude, args.width, args.random_style)
    else:
        print(f"ERROR: {args.input} not found")
        exit(1)


if __name__ == '__main__':
    main()
