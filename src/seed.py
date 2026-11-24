from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

DEFAULT_INFERENCE_SEED = 42


def seed_inference(seed: int | None) -> None:
    """Seed the inference run before model initialization and the first draw."""
    if seed is not None:
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # Seeds both CPU and CUDA generators.

# One CPU generator serves all the sampling and tie-breaking for a single ECG.
_image_generator: ContextVar[torch.Generator | None] = ContextVar("ecg_generator", default=None)


def image_generator() -> torch.Generator | None:
    return _image_generator.get()


@contextmanager
def image_random_stream(seed: int | None) -> Iterator[None]:
    import torch

    generator = None if seed is None else torch.Generator().manual_seed(seed)
    token = _image_generator.set(generator)
    try:
        yield
    finally:
        _image_generator.reset(token)
