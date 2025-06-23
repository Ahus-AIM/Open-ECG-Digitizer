import os
import random
from collections.abc import Iterable

import matplotlib.pyplot as plt
import torch
from torchvision.io import decode_image
from tqdm import tqdm
from yacs.config import CfgNode as CN

from src.config.default import get_cfg
from src.utils import load_model


def load_file_paths_with_structure(config: CN) -> Iterable[str]:
    """
    Loads file paths from the specified directory structure in the config.
    Returns an iterable of file paths.
    """
    candidate_filenames = []
    for root, _, files in os.walk(config.DATA.images_path):
        for file in files:
            if file.endswith(tuple(config.DATA.image_extensions)):
                candidate_filenames.append(os.path.join(root, file))
    random.seed(42)
    random.shuffle(candidate_filenames)
    return candidate_filenames


def copy_file_structure(src: str, dst: str) -> None:
    """
    Copies only the directory structure (not files) from src to dst.
    """
    for root, dirs, _ in os.walk(src):
        rel_path = os.path.relpath(root, src)
        target_dir = os.path.join(dst, rel_path)
        os.makedirs(target_dir, exist_ok=True)


def adjust_image_shape(image: torch.Tensor) -> torch.Tensor:
    C, H, W = image.shape
    if C == 1:
        image = image.expand(3, H, W)
    elif C == 4:
        image = image[:3]
    return image.unsqueeze(0)


def main(config: CN) -> None:
    inference_wrapper = load_model(config)

    loading_bar = tqdm(load_file_paths_with_structure(config))
    if config.DATA.get("output_path") is not None:
        if os.path.exists(config.DATA.output_path):
            for root, dirs, files in os.walk(config.DATA.output_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        if not os.path.exists(config.DATA.output_path):
            os.makedirs(config.DATA.output_path)
        copy_file_structure(config.DATA.images_path, config.DATA.output_path)

    for file_path in loading_bar:
        if "photos" not in file_path:
            continue
        image = decode_image(file_path, mode="RGB")
        image = adjust_image_shape(image)
        got_values = inference_wrapper(image)

        if config.DATA.get("output_path") is not None:
            # Compute relative path
            rel_path = os.path.relpath(file_path, config.DATA.images_path)
            output_file_path = os.path.join(config.DATA.output_path, rel_path)

            # Plot and save output
            fig, axs = plt.subplots(2, 2, figsize=(20, 14))
            axs[0, 0].imshow(got_values["image"].squeeze().permute(1, 2, 0).cpu().numpy())
            source_points = got_values["source_points"]
            axs[0, 0].scatter(source_points[:, 0].cpu().numpy(), source_points[:, 1].cpu().numpy(), s=20, c="red")
            axs[0, 1].imshow(got_values["image_aligned"].squeeze().permute(1, 2, 0).cpu().numpy())
            axs[1, 0].imshow(
                got_values["signal_probabilities_aligned"].squeeze().cpu().numpy(), interpolation="none", vmin=0, vmax=1
            )
            axs[1, 1].plot(got_values["snake"].cpu().numpy().T)
            axs[1, 1].invert_yaxis()
            axs[1, 1].set_xlim(0, got_values["signal_probabilities_aligned"].shape[1])
            axs[1, 1].set_ylim(got_values["signal_probabilities_aligned"].shape[0], 0)
            plt.tight_layout()

            # save the signal probabilities aligned as npy array in the output directory
            output_signal_probabilities_path = os.path.splitext(output_file_path)[0] + "_signal_probabilities.npy"
            os.makedirs(os.path.dirname(output_signal_probabilities_path), exist_ok=True)

            import numpy as np

            signal_probs_to_save = got_values["signal_probabilities_aligned"].squeeze().cpu().numpy()
            np.save(output_signal_probabilities_path, signal_probs_to_save)
            grid_probs_aligned = got_values["grid_probabilities_aligned"].squeeze().cpu().numpy()
            pixpermm = 2 / (got_values["mm_per_pixel_x"] + got_values["mm_per_pixel_y"])
            np.save(
                os.path.splitext(output_file_path)[0] + f"XXX{pixpermm:.3f}XXX_grid_probabilities.npy",
                grid_probs_aligned,
            )

            # a box is 5 mm, plot some boxes on the image
            for i in range(0, 15, 2):
                for j in range(0, 15, 2):
                    axs[0, 1].add_patch(
                        plt.Rectangle(  # type: ignore
                            (i * 5 / got_values["mm_per_pixel_x"], j * 5 / got_values["mm_per_pixel_y"]),
                            width=5 / got_values["mm_per_pixel_x"],
                            height=5 / got_values["mm_per_pixel_y"],
                            edgecolor="red",
                            facecolor="none",
                        )
                    )

            # Change file extension to .png for saving
            output_file_path = os.path.splitext(output_file_path)[0] + ".png"
            plt.savefig(output_file_path, bbox_inches="tight", dpi=150)
            plt.close()


if __name__ == "__main__":
    cfg = get_cfg("src/config/inference_wrapper.yml")
    main(cfg)
