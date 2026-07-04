#!/usr/bin/env python3
"""Multi-layout ECG renderer backed by DeepECG_Preprocess.

This module preserves the Open-ECG-Digitizer rendering script API while using
``ecg_plotter`` from HeartWise-AI/DeepECG_Preprocess for the actual layout
catalogue and rendering implementation. Keeping this wrapper lets existing
training scripts import ``LAYOUTS``, ``load_ecg_npy``, ``render_ecg``, and
``render_ecg_with_mask`` without carrying a second renderer implementation.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


try:
    from ecg_plotter import LAYOUTS, LEAD_NAMES, render_lead_dict, render_lead_dict_with_mask
except ModuleNotFoundError:
    sibling = Path(__file__).resolve().parents[2] / "DeepECG_Preprocess"
    if sibling.exists():
        sys.path.insert(0, str(sibling))
    from ecg_plotter import LAYOUTS, LEAD_NAMES, render_lead_dict, render_lead_dict_with_mask


def load_ecg_npy(npy_path: str) -> dict:
    """Load an ECG NPY into a lead dictionary for the shared renderer.

    Handles shapes: ``(2500, 12)``, ``(2500, 12, 1)``, and ``(12, 2500)``.
    The expected column order follows DeepECG_Preprocess/MHI convention.
    """
    data = np.squeeze(np.load(npy_path))
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    if data.shape[0] == len(LEAD_NAMES) and data.shape[1] != len(LEAD_NAMES):
        data = data.T
    if data.shape[1] < len(LEAD_NAMES):
        raise ValueError(
            f"Expected at least {len(LEAD_NAMES)} leads, got shape {data.shape}"
        )

    lead_dict = {name: data[:, idx] for idx, name in enumerate(LEAD_NAMES)}
    lead_dict["-aVR"] = -lead_dict["aVR"]
    return lead_dict


def render_ecg(
    lead_dict: dict,
    layout_name: str = "standard_3x4_with_r1",
    amplitude_factor: float = 4.88,
    width: int = 2500,
    grid_color: str = "red",
    grid_alpha: float = 1.0,
    show_labels: bool = True,
    show_grid: bool = True,
    linewidth: float = 3.0,
    line_color: str = "#000000",
    title: str = "",
    random_rhythm: bool = False,
    label_fontsize: int = 28,
    label_color: str = "black",
    background_color: str = "white",
    mask_mode: bool = False,
    _resolved_rhythm: list = None,
) -> Image.Image:
    """Render ECG leads into a PIL image using DeepECG_Preprocess layouts."""
    return render_lead_dict(
        lead_dict,
        layout_name=layout_name,
        amplitude_factor=amplitude_factor,
        width=width,
        grid_color=grid_color,
        grid_alpha=grid_alpha,
        show_labels=show_labels,
        show_grid=show_grid,
        linewidth=linewidth,
        line_color=line_color,
        title=title,
        random_rhythm=random_rhythm,
        label_fontsize=label_fontsize,
        label_color=label_color,
        background_color=background_color,
        mask_mode=mask_mode,
        resolved_rhythm=_resolved_rhythm,
    )


def render_ecg_with_mask(
    lead_dict: dict,
    layout_name: str = "standard_3x4_with_r1",
    amplitude_factor: float = 4.88,
    width: int = 2500,
    random_rhythm: bool = False,
    **image_kwargs,
) -> tuple:
    """Render an ECG image and matching segmentation mask.

    Mask colors follow the digitizer training convention:
    red=grid, green=text, blue=signal, black=background.
    """
    return render_lead_dict_with_mask(
        lead_dict,
        layout_name=layout_name,
        amplitude_factor=amplitude_factor,
        width=width,
        random_rhythm=random_rhythm,
        **image_kwargs,
    )


def render_ecg_random_style(
    lead_dict: dict,
    layout_name: str = "standard_3x4_with_r1",
    amplitude_factor: float = 4.88,
    width: int = 2500,
) -> Image.Image:
    """Render ECG with randomized visual style for training data diversity."""
    grid_colors = [
        "red", "#FF6B6B", "#CC0000", "#FF4444",
        "#00AA00", "#006600", "#4488FF", "#2266CC",
        "#FF8800", "#CC6600",
    ]
    return render_ecg(
        lead_dict,
        layout_name,
        amplitude_factor=amplitude_factor * random.uniform(0.85, 1.15),
        width=width,
        grid_color=random.choice(grid_colors),
        grid_alpha=random.uniform(0.5, 1.0),
        show_grid=random.random() > 0.05,
        show_labels=random.random() > 0.1,
        linewidth=random.uniform(1.5, 4.0),
        line_color=random.choice(["#000000", "#111111", "#222222", "#000044"]),
        random_rhythm=True,
        label_fontsize=random.randint(20, 36),
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
    """Batch render NPY files in multiple layouts."""
    if layouts is None:
        layouts = list(LAYOUTS.keys())

    os.makedirs(output_dir, exist_ok=True)
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])
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
                    img = render_ecg_random_style(
                        lead_dict, layout_name, amplitude_factor, width
                    )
                else:
                    img = render_ecg(
                        lead_dict,
                        layout_name,
                        amplitude_factor=amplitude_factor,
                        width=width,
                        random_rhythm=True,
                    )

                out_name = npy_file.replace(".npy", ".png")
                img.save(os.path.join(layout_dir, out_name))
                count += 1
                if count % 50 == 0:
                    print(f"  [{count}/{total}] {layout_name}/{out_name}")
            except Exception as exc:
                errors += 1
                if errors <= 10:
                    print(f"  ERROR {npy_file}: {exc}")

    print(f"\nDone: {count}/{total} images saved to {output_dir} ({errors} errors)")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Layout ECG Renderer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported layouts:
  standard_3x4          standard_3x4_with_r1   standard_3x4_with_r2
  standard_3x4_with_r3  standard_6x2           standard_6x2_with_r1
  standard_12x1         cabrera_12x1           precordial_6x1
  standard_6x1_limb     cabrera_6x1_limb       standard_3x1
  precordial_3x2
""",
    )
    parser.add_argument("input", help="NPY file or folder of NPYs")
    parser.add_argument("--output", "-o", default="rendered_ecgs", help="Output directory")
    parser.add_argument(
        "--layout",
        "-l",
        default="standard_3x4_with_r1",
        choices=list(LAYOUTS.keys()),
        help="Layout (single file mode)",
    )
    parser.add_argument("--all-layouts", action="store_true", help="Render all layouts")
    parser.add_argument("--layouts", nargs="+", choices=list(LAYOUTS.keys()))
    parser.add_argument("--n-per-layout", type=int, default=None)
    parser.add_argument("--width", type=int, default=2500)
    parser.add_argument("--amplitude", type=float, default=4.88)
    parser.add_argument("--random-style", action="store_true")
    parser.add_argument("--random-rhythm", action="store_true")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        lead_dict = load_ecg_npy(args.input)
        layouts = (
            list(LAYOUTS.keys())
            if args.all_layouts
            else (args.layouts or [args.layout])
        )
        os.makedirs(args.output, exist_ok=True)
        basename = os.path.splitext(os.path.basename(args.input))[0]

        for layout_name in layouts:
            if args.random_style:
                img = render_ecg_random_style(
                    lead_dict, layout_name, args.amplitude, args.width
                )
            else:
                img = render_ecg(
                    lead_dict,
                    layout_name,
                    args.amplitude,
                    args.width,
                    random_rhythm=args.random_rhythm,
                )

            out_path = os.path.join(args.output, f"{basename}_{layout_name}.png")
            img.save(out_path)
            print(f"Saved: {out_path} ({img.size[0]}x{img.size[1]})")

    elif os.path.isdir(args.input):
        layouts = list(LAYOUTS.keys()) if args.all_layouts else (args.layouts or [args.layout])
        batch_render(
            args.input,
            args.output,
            layouts,
            args.n_per_layout,
            args.amplitude,
            args.width,
            args.random_style,
        )
    else:
        print(f"ERROR: {args.input} not found")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
