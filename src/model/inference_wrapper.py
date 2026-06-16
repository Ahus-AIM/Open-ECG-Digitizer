import time
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn import Module
from yacs.config import CfgNode as CN

from src.utils import import_class_from_path


@contextmanager
def timed_section(name: str, times_dict: dict[str, float]) -> Generator[None, None, None]:
    """Context manager for timing code blocks.

    Args:
        name: Name of the section.
        times_dict: Dictionary to store timing.
    """
    start = time.time()
    yield
    times_dict[name] = time.time() - start


class InferenceWrapper(Module):
    def __init__(
        self,
        config: CN,
        device: str,
        resample_size: None | tuple[int, ...] = None,
        grid_class: int = 0,
        text_background_class: int = 1,
        signal_class: int = 2,
        background_class: int = 3,
        rotate_on_resample: bool = False,
        enable_timing: bool = False,
        minimum_image_size: int = 512,
        apply_dewarping: bool = True,
    ) -> None:
        """Inference wrapper for ECG pipeline.

        Args:
            config: Configuration node.
            device: Torch device string.
            resample_size: Optional resample target size.
            grid_class: Grid class index.
            text_background_class: Text and background class index.
            signal_class: Signal class index.
            background_class: Background class index.
            rotate_on_resample: Whether to rotate on resample.
            enable_timing: Whether to print timings.
            minimum_image_size: Minimum allowed image size.
            apply_dewarping: Whether to apply dewarping (perspective correction is still performed regardless).
        """
        super().__init__()
        self.config = config
        self.device = device
        self.resample_size = resample_size
        self.grid_class = grid_class
        self.text_background_class = text_background_class
        self.signal_class = signal_class
        self.background_class = background_class
        self.rotate_on_resample = rotate_on_resample
        self._timing_enabled = enable_timing
        self.minimum_image_size = minimum_image_size
        self.apply_dewarping = apply_dewarping

        self.signal_extractor = self._load_signal_extractor()
        self.perspective_detector: Any = self._load_perspective_detector()
        self.segmentation_model: Any = self._load_segmentation_model().to(self.device)
        self.cropper: Any = self._load_cropper()
        self.pixel_size_finder: Any = self._load_pixel_size_finder()
        self.dewarper: Any = self._load_dewarper()
        self.identifier = self._load_layout_identifier()
        self.times: dict[str, float] = {}

    @torch.no_grad()
    def forward(
        self, image: Tensor, layout_should_include_substring: None | str
    ) -> dict[str, Tensor | str | float | None | dict[str, Any]]:
        """Performs full inference on an input image.

        Args:
            image: Input image tensor.
            layout_should_include_substring: Optional substring to filter layout names.

        Returns:
            Dictionary with processed outputs and intermediate results.
        """
        self._check_image_dimensions(image)
        image = self.min_max_normalize(image)
        image = image.to(self.device)

        self.times = {}
        image = self._resample_image(image)

        signal_prob, grid_prob, text_prob = self._get_feature_maps(image)

        # Perspective + crop with a full-image fallback: zoomed-in photos may
        # lack visible paper borders, which makes perspective detection or
        # cropping raise. Falling back to the full-image rectangle yields a
        # usable (if uncorrected) signal instead of a hard crash.
        self.bounding_box_used = True
        try:
            with timed_section("Perspective detection", self.times):
                alignment_params = self.perspective_detector(grid_prob)
            with timed_section("Cropping", self.times):
                source_points = self.cropper(signal_prob, alignment_params)
        except Exception:
            self.bounding_box_used = False
            H, W = image.shape[-2], image.shape[-1]
            source_points = torch.tensor(
                [[0.0, 0.0], [float(W), 0.0], [float(W), float(H)], [0.0, float(H)]],
                dtype=torch.float32,
                device=image.device,
            )

        aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._align_feature_maps(
            image, signal_prob, grid_prob, text_prob, source_points
        )

        with timed_section("Pixel size search", self.times):
            mm_per_pixel_x, mm_per_pixel_y = self.pixel_size_finder(aligned_grid_prob)
            avg_pixel_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2

        with timed_section("Dewarping", self.times):
            if self.apply_dewarping:
                self.dewarper.fit(aligned_grid_prob.squeeze(), avg_pixel_per_mm)
                aligned_signal_prob = self.dewarper.transform(aligned_signal_prob.squeeze())

        with timed_section("Signal extraction", self.times):
            signals = self.signal_extractor(aligned_signal_prob.squeeze())

        self._print_profiling_results()

        layout = self.identifier(
            signals,
            aligned_text_prob,
            avg_pixel_per_mm,
            layout_should_include_substring=layout_should_include_substring,
        )
        try:
            layout_str = layout["layout"]
            layout_is_flipped = str(layout["flip"])
            layout_cost = layout.get("cost", 1.0)
        except KeyError:
            layout_str = "Unknown layout"
            layout_is_flipped = "False"
            layout_cost = 1.0

        return {
            "layout_name": layout_str,
            "input_image": image.cpu(),
            "aligned": {
                "image": aligned_image.cpu(),
                "signal_prob": aligned_signal_prob.cpu(),
                "grid_prob": aligned_grid_prob.cpu(),
                "text_prob": aligned_text_prob.cpu(),
            },
            "signal": {
                "raw_lines": signals.cpu(),
                "canonical_lines": layout.get("canonical_lines", None),
                "lines": layout.get("lines", None),
                "layout_matching_cost": layout_cost,
                "layout_is_flipped": layout_is_flipped,
            },
            "pixel_spacing_mm": {
                "x": mm_per_pixel_x,
                "y": mm_per_pixel_y,
                "average_pixel_per_mm": avg_pixel_per_mm,
            },
            "source_points": source_points.cpu(),
            "bounding_box_used": self.bounding_box_used,
        }

    def _align_feature_maps(
        self,
        image: Tensor,
        signal_prob: Tensor,
        grid_prob: Tensor,
        text_prob: Tensor,
        source_points: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Aligns image and feature maps using perspective cropping.

        Returns:
            Aligned image, signal, grid, and text tensors.
        """
        with timed_section("Feature map resampling", self.times):
            aligned_signal_prob = self.cropper.apply_perspective(signal_prob, source_points, fill_value=0)
            aligned_image = self.cropper.apply_perspective(image, source_points, fill_value=0)
            aligned_grid_prob = self.cropper.apply_perspective(grid_prob, source_points, fill_value=0)
            aligned_text_prob = self.cropper.apply_perspective(text_prob, source_points, fill_value=0)
            if self.rotate_on_resample:
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._rotate_on_resample(
                    aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
                )
            aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._crop_y(
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
            )

            return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _crop_y(
        self, image: Tensor, signal_prob: Tensor, grid_prob: Tensor, text_prob: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Crops tensors in y and x using bounds from feature maps.

        Returns:
            Cropped image, signal, grid, and text tensors.
        """

        def get_bounds(tensor: Tensor) -> tuple[int, int]:
            prob = torch.clamp(
                tensor.squeeze().sum(dim=tensor.dim() - 3) - tensor.squeeze().sum(dim=tensor.dim() - 3).mean(),
                min=0,
            )
            non_zero = (prob > 0).nonzero(as_tuple=True)[0]
            if non_zero.numel() == 0:
                return 0, tensor.shape[2] - 1
            return int(non_zero[0].item()), int(non_zero[-1].item())

        y1, y2 = get_bounds(signal_prob + grid_prob)

        slices = (slice(None), slice(None), slice(y1, y2 + 1), slice(None))
        return image[slices], signal_prob[slices], grid_prob[slices], text_prob[slices]

    def _print_profiling_results(self) -> None:
        """Prints the timings for each timed section."""
        if not self._timing_enabled:
            return
        print(" Timing results:")
        max_length = max(len(section) for section in self.times.keys())
        for section, duration in self.times.items():
            print(f"    {section:<{max_length+2}}{duration:.2f} s")
        total_time = sum(self.times.values())
        print(f"Total time: {total_time:.2f} s")

    def _rotate_on_resample(
        self,
        aligned_image: Tensor,
        aligned_signal_prob: Tensor,
        aligned_grid_prob: Tensor,
        aligned_text_prob: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Rotates all tensors if height > width.

        Returns:
            Rotated tensors in same order.
        """
        if aligned_image.shape[2] > aligned_image.shape[3]:
            aligned_image = torch.rot90(aligned_image, k=3, dims=(2, 3))
            aligned_signal_prob = torch.rot90(aligned_signal_prob, k=3, dims=(2, 3))
            aligned_grid_prob = torch.rot90(aligned_grid_prob, k=3, dims=(2, 3))
            aligned_text_prob = torch.rot90(aligned_text_prob, k=3, dims=(2, 3))
        return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _resample_image(self, image: Tensor) -> Tensor:
        with timed_section("Initial resampling", self.times):
            if self.resample_size is None:
                return image

            height, width = image.shape[2], image.shape[3]
            min_dim = min(height, width)
            max_dim = max(height, width)

            if min_dim < self.minimum_image_size:
                scale: float = self.minimum_image_size / min_dim
                new_size: tuple[int, int] = (int(height * scale), int(width * scale))
                interpolated: Tensor = F.interpolate(image, size=new_size, mode="bilinear", align_corners=False)
                return interpolated

            if isinstance(self.resample_size, int):
                if max_dim > self.resample_size:
                    scale = self.resample_size / max_dim
                    new_size = (int(height * scale), int(width * scale))
                    return F.interpolate(image, size=new_size, mode="bilinear", align_corners=False, antialias=True)
                return image

            if isinstance(self.resample_size, tuple):
                interpolated = F.interpolate(
                    image, size=self.resample_size, mode="bilinear", align_corners=False, antialias=True
                )
                return interpolated

            raise ValueError(f"Invalid resample_size: {self.resample_size}. Expected int or tuple of (height, width).")

    def process_sparse_prob(self, signal_prob: Tensor) -> Tensor:
        signal_prob = signal_prob - signal_prob.mean() * 1
        signal_prob = torch.clamp(signal_prob, min=0)
        signal_prob = signal_prob / (signal_prob.max() + 1e-9)
        return signal_prob

    def _get_feature_maps(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        with timed_section("Segmentation", self.times):
            logits = self.segmentation_model(image)
            prob = torch.softmax(logits, dim=1)

            signal_prob = prob[:, [self.signal_class], :, :]
            grid_prob = prob[:, [self.grid_class], :, :]
            text_prob = prob[:, [self.text_background_class], :, :]

            signal_prob = self.process_sparse_prob(signal_prob)
            grid_prob = self.process_sparse_prob(grid_prob)
            text_prob = self.process_sparse_prob(text_prob)

            return signal_prob, grid_prob, text_prob

    def min_max_normalize(self, image: Tensor) -> Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def _load_signal_extractor(self) -> Any:
        signal_extractor_class = import_class_from_path(self.config.SIGNAL_EXTRACTOR.class_path)
        extractor: Any = signal_extractor_class(**self.config.SIGNAL_EXTRACTOR.KWARGS)
        return extractor

    def _load_perspective_detector(self) -> Any:
        perspective_detector_class = import_class_from_path(self.config.PERSPECTIVE_DETECTOR.class_path)
        perspective_detector: Any = perspective_detector_class(**self.config.PERSPECTIVE_DETECTOR.KWARGS)
        return perspective_detector

    def _load_segmentation_model(self) -> Any:
        segmentation_model_class = import_class_from_path(self.config.SEGMENTATION_MODEL.class_path)
        segmentation_model: Any = segmentation_model_class(**self.config.SEGMENTATION_MODEL.KWARGS)
        self._load_segmentation_model_weights(segmentation_model)
        return segmentation_model.eval()

    def _load_cropper(self) -> Any:
        cropper_class = import_class_from_path(self.config.CROPPER.class_path)
        cropper: Any = cropper_class(**self.config.CROPPER.KWARGS)
        return cropper

    def _load_pixel_size_finder(self) -> Any:
        pixel_size_finder_class = import_class_from_path(self.config.PIXEL_SIZE_FINDER.class_path)
        pixel_size_finder: Any = pixel_size_finder_class(**self.config.PIXEL_SIZE_FINDER.KWARGS)
        return pixel_size_finder

    def _load_dewarper(self) -> Any:
        dewarper_class = import_class_from_path(self.config.DEWARPER.class_path)
        dewarper: Any = dewarper_class(**self.config.DEWARPER.KWARGS)
        return dewarper

    def _load_layout_identifier(self) -> Any:
        layouts = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.config_path, "r"))
        unet_cfg = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.unet_config_path, "r"))
        unet_class = import_class_from_path(unet_cfg["MODEL"]["class_path"])
        unet: torch.nn.Module = unet_class(**unet_cfg["MODEL"]["KWARGS"])
        checkpoint = torch.load(self.config.LAYOUT_IDENTIFIER.unet_weight_path, map_location=self.device)
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        unet.load_state_dict(checkpoint)
        unet.eval()

        identifier_class = import_class_from_path(self.config.LAYOUT_IDENTIFIER.class_path)
        identifier: Any = identifier_class(
            layouts=layouts,
            unet=unet,
            **self.config.LAYOUT_IDENTIFIER.KWARGS,
        )
        return identifier

    def _load_segmentation_model_weights(self, segmentation_model: torch.nn.Module) -> None:
        """Loads weights for segmentation model.

        Args:
            segmentation_model: The model to load weights into.
        """
        checkpoint = torch.load(self.config.SEGMENTATION_MODEL.weight_path, weights_only=True, map_location=self.device)
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        segmentation_model.load_state_dict(checkpoint)

    def _check_image_dimensions(self, image: Tensor) -> None:
        """Checks input image dimensions.

        Args:
            image: Image tensor.

        Raises:
            NotImplementedError: If batch or channel dims are incorrect.
        """
        if image.dim() != 4:
            raise NotImplementedError(f"Expected 4 dimensions, got tensor with {image.dim()} dimensions")
        if image.shape[0] != 1:
            raise NotImplementedError(f"Batch processing not supported, got tensor with shape {image.shape}")
        if image.shape[1] != 3:
            raise NotImplementedError(f"Expected 3 channels, got tensor with shape {image.shape}")
