import torch
from src.utils import import_class_from_path
from typing import Dict, Tuple
from yacs.config import CfgNode as CN


class InferenceWrapper(torch.nn.Module):
    def __init__(
        self,
        config: CN,
        device: str,
        resample_size: None | Tuple[int] = None,
        signal_class: int = 2,
        background_class: int = 0,
    ) -> None:
        super(InferenceWrapper, self).__init__()
        self.config = config
        self.device = device
        self.resample_size = resample_size
        self.signal_class = signal_class
        self.background_class = background_class
        self.snake = self._load_snake()
        self.perspective_detector = self._load_perspective_detector()
        self.segmentation_model = self._load_segmentation_model().to(self.device)
        self.cropper = self._load_cropper()
        self.pixel_size_finder = self._load_pixel_size_finder()

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._check_image_dimensions(image)
        image = self.min_max_normalize(image)
        image = image.to(self.device)

        if self.resample_size is not None:
            image = torch.nn.functional.interpolate(image, size=self.resample_size)

        signal_probabilities, background_probabilities = self.segment_signal_and_background(image)
        alignment_params = self.perspective_detector(image)

        source_points = self.cropper(signal_probabilities, alignment_params)

        mean_image_color = image.mean(dim=(0, 2, 3))
        aligned_signal_probabilities = self.cropper.apply_perspective(signal_probabilities, source_points, fill_value=0)
        aligned_background_probabilities = self.cropper.apply_perspective(
            background_probabilities, source_points, fill_value=1
        )
        aligned_image = self.cropper.apply_perspective(image, source_points, fill_value=mean_image_color.tolist())
        mean_background = mean_image_color.repeat(aligned_image.shape[-2], aligned_image.shape[-1], 1).permute(2, 0, 1)
        aligned_background = (
            aligned_background_probabilities * aligned_image + (1 - aligned_background_probabilities) * mean_background
        )

        mm_per_pixel_x, mm_per_pixel_y = self.pixel_size_finder(aligned_background)

        self.snake.fit(aligned_signal_probabilities.squeeze().cpu())

        return {
            "image": image.cpu(),
            "image_aligned": aligned_image.cpu(),
            "signal_probabilities_aligned": aligned_signal_probabilities.cpu(),
            "snake": self.snake.snake.data.detach(),
            "mm_per_pixel_x": mm_per_pixel_x,
            "mm_per_pixel_y": mm_per_pixel_y,
        }

    def resample(self, image: torch.Tensor) -> torch.Tensor:
        if self.resample_size is not None:
            image = torch.nn.functional.interpolate(image, size=self.resample_size)
        return image

    def segment_signal_and_background(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.segmentation_model(image)
        probabilities = torch.softmax(logits, dim=1)
        signal_prob = probabilities[:, self.signal_class : self.signal_class + 1, :, :]
        background_prob = probabilities[:, self.background_class : self.background_class + 1, :, :]
        return signal_prob, background_prob

    def min_max_normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def _load_snake(self) -> torch.nn.Module:
        snake_class = import_class_from_path(self.config.SNAKE.class_path)
        snake: torch.nn.Module = snake_class(**self.config.SNAKE.KWARGS)
        return snake

    def _load_perspective_detector(self) -> torch.nn.Module:
        perspective_detector_class = import_class_from_path(self.config.PERSPECTIVE_DETECTOR.class_path)
        perspective_detector: torch.nn.Module = perspective_detector_class(**self.config.PERSPECTIVE_DETECTOR.KWARGS)
        return perspective_detector

    def _load_segmentation_model(self) -> torch.nn.Module:
        segmentation_model_class = import_class_from_path(self.config.SEGMENTATION_MODEL.class_path)
        segmentation_model: torch.nn.Module = segmentation_model_class(**self.config.SEGMENTATION_MODEL.KWARGS)
        self._load_segmentation_model_weights(segmentation_model)
        return segmentation_model

    def _load_cropper(self) -> torch.nn.Module:
        cropper_class = import_class_from_path(self.config.CROPPER.class_path)
        cropper: torch.nn.Module = cropper_class(**self.config.CROPPER.KWARGS)
        return cropper

    def _load_pixel_size_finder(self) -> torch.nn.Module:
        pixel_size_finder_class = import_class_from_path(self.config.PIXEL_SIZE_FINDER.class_path)
        pixel_size_finder: torch.nn.Module = pixel_size_finder_class(**self.config.PIXEL_SIZE_FINDER.KWARGS)
        return pixel_size_finder

    def _load_segmentation_model_weights(self, segmentation_model: torch.nn.Module) -> None:
        checkpoint = torch.load(
            self.config.SEGMENTATION_MODEL.weight_path, weights_only=True, map_location=self.device
        )[0]
        segmentation_model.load_state_dict(checkpoint)

    def _check_image_dimensions(self, image: torch.Tensor) -> None:
        if image.dim() != 4:
            raise NotImplementedError(f"Expected 4 dimensions, got tensor with {image.dim()} dimensions")
        if image.shape[0] != 1:
            raise NotImplementedError(f"Batch processing not supported, got tensor with shape {image.shape}")
        if image.shape[1] != 3:
            raise NotImplementedError(f"Expected 3 channels, got tensor with shape {image.shape}")
