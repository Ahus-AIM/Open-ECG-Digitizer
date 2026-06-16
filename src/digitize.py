import argparse
import os
import random
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
from yacs.config import CfgNode as CN

from src.config.default import get_cfg
from src.utils import find_config_path, import_class_from_path

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_candidate_file_paths(config: CN) -> list[str]:
    candidate_filenames: list[str] = []
    for root, _, files in os.walk(config.DATA.images_path):
        for file in files:
            if file.endswith(tuple(config.DATA.image_extensions)):
                candidate_filenames.append(os.path.join(root, file))
    random.seed(42)
    random.shuffle(candidate_filenames)
    return candidate_filenames


def clear_and_prepare_output_dir(config: CN) -> None:
    output_path: str = config.DATA.output_path
    if os.path.exists(output_path):
        for root, dirs, files in os.walk(output_path, topdown=False):
            if config.DATA.get("clear_output_dir_if_exists", False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_file_structure(config.DATA.images_path, output_path)


def copy_file_structure(src: str, dst: str) -> None:
    for root, dirs, _ in os.walk(src):
        rel_path: str = os.path.relpath(root, src)
        target_dir: str = os.path.join(dst, rel_path)
        os.makedirs(target_dir, exist_ok=True)


def decode_and_prepare_image(file_path: str) -> torch.Tensor:
    image: torch.Tensor = read_image(file_path, mode=ImageReadMode.RGB)
    C, H, W = image.shape
    if C == 1:
        image = image.expand(3, H, W)
    elif C == 4:
        image = image[:3]
    return image.unsqueeze(0)


def canonical_from_got_values(got_values: dict[str, Any]) -> torch.Tensor | None:
    canonical: torch.Tensor | None = None
    if "signal" in got_values and isinstance(got_values["signal"], dict):
        canonical = got_values["signal"].get("canonical_lines")
    elif "canonical_lines" in got_values:
        canonical = got_values["canonical_lines"]
    return canonical


def save_timeseries_csv(canonical: torch.Tensor | None, output_basepath: str) -> None:
    if canonical is None:
        return
    data: npt.NDArray[Any] = canonical.squeeze().cpu().numpy()
    # Assume shape (n_leads, n_points)
    if data.ndim == 1:
        data = data[None, :]
    n_leads = data.shape[0]
    col_names = LEAD_NAMES[:n_leads]
    # Transpose to shape (n_points, n_leads)
    data = data.T
    header = ",".join(col_names)
    np.savetxt(output_basepath + "_timeseries_canonical.csv", data, delimiter=",", header=header, comments="")


def save_png_plot(got_values: dict[str, Any], canonical: torch.Tensor | None, output_basepath: str) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    axs[0, 0].imshow(got_values["input_image"].squeeze().permute(1, 2, 0).cpu().numpy() * 0.999)
    source_points = got_values["source_points"]
    axs[0, 0].scatter(source_points[:, 0].cpu().numpy(), source_points[:, 1].cpu().numpy(), s=20, c="red")
    axs[0, 1].imshow(got_values["aligned"]["image"].squeeze().permute(1, 2, 0).cpu().numpy() * 0.999)
    axs[1, 0].imshow(got_values["aligned"]["signal_prob"].squeeze().cpu().numpy(), interpolation="none", vmin=0, vmax=1)
    for i in range(0, 15, 2):
        for j in range(0, 15, 2):
            xval = i * 5 / got_values["pixel_spacing_mm"]["x"]
            yval = j * 5 / got_values["pixel_spacing_mm"]["y"]
            axs[0, 1].add_patch(
                plt.Rectangle(  # type: ignore
                    (xval, yval),
                    width=5 / got_values["pixel_spacing_mm"]["x"],
                    height=5 / got_values["pixel_spacing_mm"]["y"],
                    edgecolor="red",
                    facecolor="none",
                )
            )
    if canonical is not None:
        lines: npt.NDArray[Any] = canonical.squeeze().cpu().numpy()
        lines -= np.linspace(0, 24_000, num=lines.shape[0])[:, None]  # 2 uV offset per lead
        axs[1, 1].plot(lines.T, linewidth=0.5)
    plt.tight_layout()
    plt.suptitle(
        got_values.get("layout_name", "") + " Layout cost: " + f'{got_values["signal"]["layout_matching_cost"]:.2f}',
        fontsize=16,
    )
    plt.savefig(output_basepath + ".png", dpi=200)
    plt.close()


def save_matching_cost(got_values: dict[str, Any], output_basepath: str, status: str = "ok") -> None:
    # Metadata CSV: one row per processed file with matching cost, flip flag,
    # identified layout, the bounding-box-used flag, and a status column
    # (ok / no_canonical_lines / error). The status column lets downstream
    # consumers filter out rows where the CSV was never written.
    metadata_file = os.path.join(os.path.dirname(output_basepath), "digitization_metadata.csv")
    if not os.path.exists(metadata_file):
        with open(metadata_file, "w") as f:
            f.write("file_path,matching_cost,is_flipped,lead_layout,bounding_box_used,status\n")
    with open(metadata_file, "a") as f:
        file_name = os.path.basename(output_basepath)
        matching_cost = got_values.get("signal", {}).get("layout_matching_cost", float("nan"))
        is_flipped = got_values.get("is_flipped", got_values.get("signal", {}).get("layout_is_flipped", False))
        lead_layout = got_values.get("layout_name", "")
        bbox_used = got_values.get("bounding_box_used", True)
        f.write(f"{file_name},{matching_cost},{is_flipped},{lead_layout},{bbox_used},{status}\n")


def save_outputs(got_values: dict[str, Any], output_basepath: str, save_mode: str = "all") -> None:
    canonical = canonical_from_got_values(got_values)
    status = "ok" if canonical is not None else "no_canonical_lines"
    if save_mode in ["all", "timeseries_only"]:
        save_timeseries_csv(canonical, output_basepath)
    if save_mode in ["all", "png_only"]:
        save_png_plot(got_values, canonical, output_basepath)
    save_matching_cost(got_values, output_basepath, status=status)


def process_one_file(file_path: str, config: CN, inference_wrapper: Any, save_mode: str) -> None:
    image = decode_and_prepare_image(file_path)

    # Precedence for layout-hint substring (passed to LeadIdentifier to
    # constrain candidate layouts):
    #   1. DATA.layout_should_include_substring as a literal string  -> use it directly
    #      (e.g. "standard_3x4_with_r3" from an upstream classifier).
    #   2. DATA.layout_should_include_substring == "auto_from_filename" -> use
    #      the legacy filename heuristic ("limb" / "precordial" substring).
    #   3. Any other value / null -> no hint (all candidate layouts considered).
    hint_cfg = config.DATA.get("layout_should_include_substring")
    layout_should_include_substring: str | None = None
    if isinstance(hint_cfg, str) and hint_cfg:
        if hint_cfg == "auto_from_filename":
            if "limb" in str(file_path):
                layout_should_include_substring = "limb"
            elif "precordial" in str(file_path):
                layout_should_include_substring = "precordial"
        else:
            layout_should_include_substring = hint_cfg

    got_values = inference_wrapper(image, layout_should_include_substring=layout_should_include_substring)

    if config.DATA.get("output_path") is not None:
        rel_path = os.path.relpath(file_path, config.DATA.images_path)
        output_file_path = os.path.join(config.DATA.output_path, rel_path)
        output_basepath = os.path.splitext(output_file_path)[0]
        os.makedirs(os.path.dirname(output_basepath), exist_ok=True)
        save_outputs(got_values, output_basepath, save_mode)


def main(config: CN) -> None:
    inference_wrapper_class = import_class_from_path(config.MODEL.class_path)
    inference_wrapper = inference_wrapper_class(**config.MODEL.KWARGS)
    save_mode: str = getattr(config.DATA, "save_mode", "all")
    file_paths: list[str] = get_candidate_file_paths(config)

    include_list = config.DATA.get("path_should_include", [])

    if config.DATA.get("output_path") is not None:
        clear_and_prepare_output_dir(config)

    for file_path in tqdm(file_paths):
        should_be_included = not bool(include_list)
        for include in include_list:
            if include in file_path:
                should_be_included = True
        if not should_be_included:
            continue
        print(f"Processing file: {file_path}")
        try:
            process_one_file(file_path, config, inference_wrapper, save_mode)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digitize ECGs and save output.")
    parser.add_argument(
        "--config",
        type=str,
        help="Config file name or path (searched in . and src/config/)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override config options like A.B.C=123 or DATA.save_mode='png_only' (spaces require quotes).",
    )
    args = parser.parse_args()

    config_path = find_config_path(args.config)
    cfg = get_cfg(config_path)

    if args.overrides:
        kv_list: list[str] = []
        for ov in args.overrides:
            if "=" not in ov:
                raise ValueError(f"Malformed override '{ov}'. Use KEY=VALUE (e.g., MODEL.device='cuda:0').")
            k, v = ov.split("=", 1)

            # Check if key exists in config before merging
            node = cfg
            key_parts = k.split(".")
            exists = True
            for part in key_parts[:-1]:
                if part in node:
                    node = node[part]
                else:
                    exists = False
                    break
            if not exists or key_parts[-1] not in node:
                warnings.warn(f"Config key '{k}' not found in loaded config. Override skipped.")
                continue

            kv_list.extend([k, v])

        if kv_list:
            cfg.merge_from_list(kv_list)

    main(cfg)
