"""Overlap the two GPU networks with independent ECG tracing on CPU workers."""

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from src.model.cropper import Cropper
from src.model.inference_wrapper import InferenceWrapper
from src.model.lead_identifier import LeadIdentifier
from src.model.perspective_detector import PerspectiveDetector
from src.model.pixel_size_finder import PixelSizeFinder
from src.model.signal_extractor import SignalExtractor
from src.model.unet import UNet
from src.seed import image_generator, image_random_stream


class TraceInputs(NamedTuple):
    signal: npt.NDArray[Any]
    projections: tuple[npt.NDArray[Any], npt.NDArray[Any]]
    rng_state: npt.NDArray[np.uint8]


LeadPoints = list[tuple[str, float, float]]
TraceResult = tuple[npt.NDArray[Any], npt.NDArray[Any] | float]


def supports_signal_batch(model: Any) -> bool:
    return (
        type(model) is InferenceWrapper
        and model.seed is not None
        and not model.apply_dewarping
        and not model._timing_enabled
        and type(model.pixel_size_finder) is PixelSizeFinder
        and type(model.signal_extractor) is SignalExtractor
        and type(model.cropper) is Cropper
        and type(model.perspective_detector) is PerspectiveDetector
        and type(model.segmentation_model) is UNet
        and type(model.identifier) is LeadIdentifier
        and type(model.identifier.unet) is UNet
    )


@torch.no_grad()
def prepare_record(model: InferenceWrapper, image: torch.Tensor) -> tuple[TraceInputs, LeadPoints]:
    with image_random_stream(model.seed):
        _, _, _, signal, grid, text = model._aligned_maps(image)
        projections = model.pixel_size_finder.projections(grid)
        generator = image_generator()
        assert generator is not None
        inputs = TraceInputs(signal.squeeze().cpu().numpy(), projections, generator.get_state().numpy())
        detected = model.identifier.detect_leads(text)
    return inputs, detected


def extract_record(extractor: SignalExtractor, finder: PixelSizeFinder, inputs: TraceInputs) -> TraceResult:
    # Resume this ECG's generator immediately after its Hough samples.
    with image_random_stream(torch.from_numpy(inputs.rng_state)), torch.no_grad():
        mm_x, mm_y = finder.from_projections(*inputs.projections)
        spacing = (1 / mm_x + 1 / mm_y) / 2
        lines = extractor(torch.from_numpy(inputs.signal))
        return lines.numpy(), spacing.numpy() if isinstance(spacing, torch.Tensor) else spacing


def finish_record(
    model: InferenceWrapper, result: TraceResult, detected: LeadPoints, hint: str | None
) -> dict[str, Any]:
    lines, spacing = result
    scale = torch.from_numpy(spacing) if isinstance(spacing, np.ndarray) else spacing
    layout = model.identifier.identify_lines(
        torch.from_numpy(lines), scale, detected, layout_should_include_substring=hint
    )
    name, _, cost = model.layout_summary(layout)
    return {
        "layout_name": name,
        "signal": {"canonical_lines": layout.get("canonical_lines"), "layout_matching_cost": cost},
    }
