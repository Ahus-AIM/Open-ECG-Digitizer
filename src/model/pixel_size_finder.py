from typing import Any, Dict, Tuple, Optional
import os
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import torch


class PixelSizeFinder(torch.nn.Module):
    def __init__(
        self,
        mm_between_grid_lines: int = 50,
        samples: int = 500,
        min_number_of_grid_lines: int = 8,
        max_number_of_grid_lines: int = 120,
        max_zoom: int = 10,
        zoom_factor: float = 10.0,
        lower_grid_line_factor: float = 0.5,
        plot: bool = False,
    ):
        super(PixelSizeFinder, self).__init__()
        self.mm_between_grid_lines = mm_between_grid_lines
        self.samples = samples
        self.min_number_of_grid_lines = min_number_of_grid_lines
        self.max_number_of_grid_lines = max_number_of_grid_lines
        self.max_zoom = max_zoom
        self.zoom_factor = zoom_factor
        self.lower_grid_line_factor = lower_grid_line_factor
        self.plot = plot

    def forward(self, image: torch.Tensor, name: str = "test.png") -> Tuple[float, float]:
        """
        Finds mm/pixel from an image with a grid and an similarily colored background. The grid needs to be orthogonal
        to the rows and columns.

        Args:
            image (torch.Tensor): Image with a grid that is orthogonal to the rows and columns
            name (str): Name of the image to save.

        Returns:
            Tuple[float, float]: mm/pixel in x and y direction, respectively.
        """

        image = image.clone()
        if image.dim() > 3:
            image = image.squeeze()

        axs = None
        if self.plot:
            layout = [[0, 0], [1, 2], [3, 4]]
            _, axs = plt.subplot_mosaic(layout, figsize=(20, 10))  # type: ignore

        image = torchvision.transforms.Grayscale(1)(image)

        estimate_y = self._mm_per_pixel_y(image, axs, 0)
        image = image.swapaxes(1, 2)
        estimate_x = self._mm_per_pixel_y(image, axs, 1)
        image = image.swapaxes(1, 2)

        if self.plot:
            dir = "plots"
            if not os.path.exists(dir):
                os.makedirs(dir)
            axs[0].imshow(image.permute(1, 2, 0).numpy() / 255)  # type: ignore
            plt.tight_layout()
            plt.savefig(f"{dir}/{name}", bbox_inches="tight")
            plt.close()

        return estimate_x, estimate_y

    def _mm_per_pixel_y(
        self, image: torch.Tensor, axs: Optional[Dict[int, matplotlib.axes._axes.Axes]], row: int = 0
    ) -> float:
        pxls_between_lines, plot_params = self._find_pxls_between_horizontal_grid_lines(image[0, :, :])

        if axs is not None:
            autocorrelation = plot_params["autocorrelation"].numpy()
            axs[row * 2 + 1].vlines(
                plot_params["idxs"].numpy()[::5], autocorrelation.min(), autocorrelation.max(), colors="r"
            )
            axs[row * 2 + 1].plot(autocorrelation)
            axs[row * 2 + 2].plot(
                plot_params["tested_grid_discretizations"], plot_params["tested_grid_discretizations_score"]
            )

        return self.mm_between_grid_lines / pxls_between_lines

    def _find_pxls_between_horizontal_grid_lines(self, image: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        col_sum = image.sum(dim=-1)
        col_sum = col_sum.mean() - col_sum
        autocorrelation = torch.fft.irfft(torch.fft.rfft(col_sum).abs())

        pxls_between_lines, plot_params = self._zoom_grid_search_min_distance(autocorrelation)

        return pxls_between_lines, plot_params

    def _zoom_grid_search_min_distance(self, autocorrelation: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """
        Finds the best pixels between lines in the autocorrelation tensor by doing a grid search over potential pixels
        distances and zooming in on the current optimum until convergence.

        Args:
            autocorrelation (torch.Tensor): autocorrelation tensor

        Returns:
            Tuple[float, Dict[str, Any]]: best pixels between lines and plot parameters
        """
        autocorrelation = autocorrelation.clone()

        clip_divisor = 4
        expected_min_grid_lines = self.min_number_of_grid_lines / clip_divisor
        expected_max_grid_lines = self.max_number_of_grid_lines / clip_divisor
        autocorrelation = autocorrelation[: autocorrelation.shape[0] // clip_divisor]
        autocorrelation = autocorrelation - autocorrelation.mean()

        min_dist_grid_lines = autocorrelation.shape[-1] / expected_max_grid_lines
        max_dist_grid_lines = autocorrelation.shape[-1] / expected_min_grid_lines

        pxls_between_lines, plot_params = self._grid_search_min_distance(
            autocorrelation, self.samples, min_dist_grid_lines, max_dist_grid_lines
        )

        for _ in range(self.max_zoom):
            w = (max_dist_grid_lines - min_dist_grid_lines) / self.zoom_factor
            min_dist_grid_lines = max(pxls_between_lines - w, min_dist_grid_lines)
            max_dist_grid_lines = min(pxls_between_lines + w, max_dist_grid_lines)
            curr_pxls_between_lines, curr_plot_params = self._grid_search_min_distance(
                autocorrelation, self.samples, min_dist_grid_lines, max_dist_grid_lines
            )
            curr_idxs = curr_plot_params["idxs"]
            idxs = plot_params["idxs"]

            pxls_between_lines = curr_pxls_between_lines
            plot_params = curr_plot_params

            if curr_idxs.shape == idxs.shape and torch.allclose(curr_idxs, idxs):
                break

        return pxls_between_lines, plot_params

    def _grid_search_min_distance(
        self, autocorrelation: torch.Tensor, samples: int, min_dist_grid_lines: float, max_dist_grid_lines: float
    ) -> Tuple[float, Dict[str, Any]]:
        """Grid searches over the pixels between peaks in the interval [min_dist_grid_lines, max_dist_grid_lines] in the autocorrelation tensor.

        Args:
            autocorrelation (torch.Tensor): autocorrelation tensor
            samples (int): number of samples in the grid search
            min_dist_grid_lines (float): minimum pixels between grid lines
            max_dist_grid_lines (float): maximum pixels between grid lines

        Returns:
            Tuple[float, Dict[str, Any]]: best pixels between peaks and plot parameters
        """

        pixels_between_grid_lines = torch.linspace(min_dist_grid_lines, max_dist_grid_lines, samples)

        best_grid_discretization_score = torch.tensor(-torch.inf)
        best_pixels_between_grid_line = 0
        plot_params = {
            "idxs": torch.Tensor([]),
            "autocorrelation": autocorrelation,
            "tested_grid_discretizations": pixels_between_grid_lines,
            "tested_grid_discretizations_score": [],
        }

        for pixels_between_grid_line in pixels_between_grid_lines:
            calculated_grid_lines = torch.zeros(autocorrelation.shape)

            indices = (
                (torch.arange(0, autocorrelation.shape[-1], pixels_between_grid_line / 5, dtype=torch.float32))
                .round()
                .to(torch.int64)
            )
            indices = indices[indices < autocorrelation.shape[-1]]

            calculated_grid_lines[indices] = self.lower_grid_line_factor
            calculated_grid_lines[indices[::5]] = 1

            grid_discretization_score = (autocorrelation * calculated_grid_lines).sum()
            plot_params["tested_grid_discretizations_score"].append(grid_discretization_score)  # type: ignore

            if best_grid_discretization_score < grid_discretization_score:
                best_grid_discretization_score = grid_discretization_score
                best_pixels_between_grid_line = pixels_between_grid_line
                plot_params["idxs"] = indices

        return best_pixels_between_grid_line, plot_params
