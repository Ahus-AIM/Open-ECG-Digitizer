from collections.abc import Iterator
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt
import torch

Scalar = float | torch.Tensor


class PixelSizeFinder(torch.nn.Module):
    def __init__(
        self,
        mm_between_grid_lines: int = 5,
        samples: int = 1000,
        min_number_of_grid_lines: int = 15,
        max_number_of_grid_lines: int = 120,
        max_zoom: int = 10,
        zoom_factor: float = 10.0,
        lower_grid_line_factor: float = 0.5,
    ):
        """
        Finds the mm/pixel in x and y direction from an image with a grid and a similarily colored background.
        The grid needs to be orthogonal to the rows and columns. For an axis, the algorithm tries <samples> different
        pixels between grid lines in the interval
        [axis_shape / <max_number_of_grid_lines>, axis_shape / <min_number_of_grid_lines>]. The optimal pixel size is
        selected and the search space is reduced to 1 / <zoom_factor> of the current interval around the current
        optimum. This is repeated <max_zoom> times or until convergence. When selecting the optimum smaller lines
        between the grid lines are weighted <lower_grid_line_factor>.

        Args:
            mm_between_grid_lines (int, optional): The number of mm between grid lines, assumed to be equal in horizontal and vertical direction.
            samples (int, optional): Number of samples of pixels between grid lines at each zoom level.
            min_number_of_grid_lines (int, optional): Minimum number of grid lines for an axis in the image.
            max_number_of_grid_lines (int, optional): Maximum number of grid lines for an axis in the image.
            max_zoom (int, optional): The maximum number of times to zoom in on the search space.
            zoom_factor (float, optional): The reduction factor of the search space for each zoom.
            lower_grid_line_factor (float, optional): The weighing of the smaller grid lines.
        """

        super(PixelSizeFinder, self).__init__()
        self.mm_between_grid_lines = mm_between_grid_lines
        self.samples = samples
        self.min_number_of_grid_lines = min_number_of_grid_lines
        self.max_number_of_grid_lines = max_number_of_grid_lines
        self.max_zoom = max_zoom
        self.zoom_factor = zoom_factor
        self.lower_grid_line_factor = lower_grid_line_factor

    def forward(self, image: torch.Tensor) -> Tuple[Scalar, Scalar]:
        """
        Finds mm/pixel from an image with a grid and an similarily colored background. The grid needs to be orthogonal
        to the rows and columns.

        Args:
            image (torch.Tensor): Image with a grid that is orthogonal to the rows and columns
            name (str): Name of the image to save.

        Returns:
            Tuple[float, float]: mm/pixel in x and y direction, respectively.
        """
        return self.from_projections(*self.projections(image))

    def projections(self, image: torch.Tensor) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        image = image.squeeze()
        assert image.ndim == 2, "Image must be 2D (H, W) tensor."

        def project(axis: torch.Tensor) -> npt.NDArray[Any]:
            profile = axis.sum(dim=-1)
            return (profile - profile.mean()).cpu().numpy()

        return project(image.T), project(image)

    def from_projections(self, x: npt.NDArray[Any], y: npt.NDArray[Any]) -> Tuple[Scalar, Scalar]:
        def spacing(profile: npt.NDArray[Any], aspect: float) -> Scalar:
            correlation = torch.from_numpy(np.correlate(profile, profile, mode="full"))
            return self.mm_between_grid_lines / self._zoom_grid_search_min_distance(correlation, aspect)

        return spacing(x, 1), spacing(y, len(x) / len(y))

    def _zoom_grid_search_min_distance(self, autocorrelation: torch.Tensor, aspect: float) -> Scalar:
        """
        Finds the best pixels between lines in the autocorrelation tensor by doing a grid search over potential pixels
        distances and zooming in on the current optimum until convergence.

        Args:
            autocorrelation (torch.Tensor): autocorrelation tensor

        Returns:
            Tuple[float, Dict[str, Any]]: best pixels between lines
        """
        autocorrelation = autocorrelation.clone()

        expected_min_grid_lines = self.min_number_of_grid_lines / aspect
        expected_max_grid_lines = self.max_number_of_grid_lines / aspect
        autocorrelation = autocorrelation[: autocorrelation.shape[0] // 2].flip(0)
        autocorrelation = autocorrelation - autocorrelation.mean()

        min_dist_grid_lines: Scalar = autocorrelation.shape[-1] / expected_max_grid_lines
        max_dist_grid_lines: Scalar = autocorrelation.shape[-1] / expected_min_grid_lines

        pxls_between_lines = self._grid_search_min_distance(
            autocorrelation, self.samples, min_dist_grid_lines, max_dist_grid_lines
        )

        for _ in range(self.max_zoom):
            w = (max_dist_grid_lines - min_dist_grid_lines) / self.zoom_factor
            lower, upper = pxls_between_lines - w, pxls_between_lines + w
            min_dist_grid_lines = min_dist_grid_lines if min_dist_grid_lines > lower else lower
            max_dist_grid_lines = max_dist_grid_lines if max_dist_grid_lines < upper else upper
            curr_pxls_between_lines = self._grid_search_min_distance(
                autocorrelation, self.samples, min_dist_grid_lines, max_dist_grid_lines
            )

            pxls_between_lines = curr_pxls_between_lines

        return curr_pxls_between_lines

    def _grid_search_min_distance(
        self, autocorrelation: torch.Tensor, samples: int, min_dist_grid_lines: Scalar, max_dist_grid_lines: Scalar
    ) -> Scalar:
        """Grid searches over the pixels between peaks in the interval [min_dist_grid_lines, max_dist_grid_lines] in the autocorrelation tensor.

        Args:
            autocorrelation (torch.Tensor): autocorrelation tensor
            samples (int): number of samples in the grid search
            min_dist_grid_lines (float): minimum pixels between grid lines
            max_dist_grid_lines (float): maximum pixels between grid lines

        Returns:
            float (float): best pixels between peaks
        """

        spacings = torch.linspace(min_dist_grid_lines, max_dist_grid_lines, samples)
        length = len(autocorrelation)
        weights = torch.tensor(np.linspace(1, 0, length), dtype=torch.float32)

        # Precompute the three possible products at each position. Preserve the
        # multiply order (grid * weight first), including signed zeros and NaNs.
        def products(value: float) -> npt.NDArray[Any]:
            return (autocorrelation * (torch.full_like(autocorrelation, value) * weights)).numpy()

        empty, minor, major = products(0), products(self.lower_grid_line_factor), products(1)
        steps = (spacings / 5).tolist()

        def indices(i: int) -> npt.NDArray[np.int64]:
            return torch.arange(0, length, steps[i], dtype=torch.float32).round().long().numpy()

        def patterns(
            first: int, last: int, left: npt.NDArray[np.int64], right: npt.NDArray[np.int64]
        ) -> Iterator[tuple[int, npt.NDArray[np.int64]]]:
            # Raw arange lengths are monotone. Equal lengths give the same SIMD
            # partition, in which every rounded tick is monotone in the step.
            if first == last or np.array_equal(left, right):
                yield first, left[left < length]
            else:
                mid = (first + last) // 2
                yield from patterns(first, mid, left, left if mid == first else indices(mid))
                yield from patterns(mid + 1, last, right if mid + 1 == last else indices(mid + 1), right)

        best_score = -float("inf")
        best_index = None
        if samples:
            for i, ticks in patterns(0, samples - 1, indices(0), indices(samples - 1)):
                candidate = empty.copy()
                candidate[ticks] = minor[ticks]
                candidate[ticks[::5]] = major[ticks[::5]]
                score = torch.from_numpy(candidate).sum().item()
                if score > best_score:
                    best_score, best_index = score, i
        return spacings[best_index] if best_index is not None else 0
