from typing import Dict, List, Tuple

import torch
from torchvision.transforms.functional import perspective


class Cropper(torch.nn.Module):
    def __init__(
        self, granularity: int = 50, percentiles: Tuple[float, float] = (0.01, 0.99), alpha: float = 0.9
    ) -> None:
        """
        The Cropper module is used to correct for perspective distortion in images, while also cropping the image to mostly include the signal.

        Args:
            granularity (int): The number of lines defining the bins in the horizontal and vertical directions.
            percentiles (Tuple[float, float]): The percentiles defining the range of signal probabilities that should be included in the output
            alpha (float): A factor to control the influence of the source points on the destination points in the perspective transformation.
        """
        super(Cropper, self).__init__()
        self.granularity = granularity
        self.percentiles = percentiles
        self.alpha = alpha

    def forward(self, signal_probabilities: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Based on the params that define the directions of all vertical and horizontal lines, this function splits the signal probabilities tensor into slices.
        The slices are the areas between each consecutive horizontal or vertical line. Bounding lines are chosen such that the sum of signal probabilities in all slices
        is within the percentiles defined in self.percentiles.

        Args:
            signal_probabilities (torch.Tensor): The signal probabilities tensor.
            params (Dict[str, torch.Tensor]): The parameters defining the directions of the lines.

        Returns:
            torch.Tensor: The source points defining the quadrilateral that contains the signal, on the form [top-left, top-right, bottom-right, bottom-left].
        """
        normalized_signal_probabilities = self._normalize_signal_probabilities(signal_probabilities)
        rhos, thetas_horizontal, thetas_vertical = self._initialize_parameters(params)

        lower_bound_horizontal, upper_bound_horizontal = self._get_bounds(
            normalized_signal_probabilities, thetas_horizontal, rhos, "horizontal"
        )
        lower_bound_vertical, upper_bound_vertical = self._get_bounds(
            normalized_signal_probabilities, thetas_vertical, rhos, "vertical"
        )

        source_points = self._calculate_source_points(
            rhos,
            thetas_horizontal,
            thetas_vertical,
            lower_bound_horizontal,
            upper_bound_horizontal,
            lower_bound_vertical,
            upper_bound_vertical,
        )
        return source_points

    def _calculate_source_points(
        self,
        rhos: torch.Tensor,
        thetas_horizontal: torch.Tensor,
        thetas_vertical: torch.Tensor,
        lower_bound_horizontal: int,
        upper_bound_horizontal: int,
        lower_bound_vertical: int,
        upper_bound_vertical: int,
    ) -> torch.Tensor:

        rhos = rhos.flip(0)
        thetas_horizontal = thetas_horizontal.flip(0)
        thetas_vertical = thetas_vertical.flip(0)

        rho_min_horizontal = rhos[lower_bound_horizontal]
        rho_max_horizontal = rhos[upper_bound_horizontal]
        theta_min_horizontal = thetas_horizontal[lower_bound_horizontal]
        theta_max_horizontal = thetas_horizontal[upper_bound_horizontal]
        rho_min_vertical = rhos[lower_bound_vertical]
        rho_max_vertical = rhos[upper_bound_vertical]
        theta_min_vertical = thetas_vertical[lower_bound_vertical]
        theta_max_vertical = thetas_vertical[upper_bound_vertical]

        # This will later be input to torchvision.transforms.functional.perspective and should have the format:
        # List containing four lists of two integers corresponding to four corners [top-left, top-right, bottom-right, bottom-left]
        return torch.stack(
            [
                self._line_intersection_from_hough(
                    rho_min_horizontal, theta_min_horizontal, rho_min_vertical, theta_min_vertical
                ),
                self._line_intersection_from_hough(
                    rho_max_horizontal, theta_max_horizontal, rho_min_vertical, theta_min_vertical
                ),
                self._line_intersection_from_hough(
                    rho_max_horizontal, theta_max_horizontal, rho_max_vertical, theta_max_vertical
                ),
                self._line_intersection_from_hough(
                    rho_min_horizontal, theta_min_horizontal, rho_max_vertical, theta_max_vertical
                ),
            ],
            dim=0,
        )

    def _get_bounds(
        self, normalized_signal_probabilities: torch.Tensor, thetas: torch.Tensor, rhos: torch.Tensor, mode: str
    ) -> Tuple[int, int]:
        buffer = self._create_buffer(normalized_signal_probabilities, thetas, rhos, mode)
        values = self._calculate_values(buffer, normalized_signal_probabilities)
        return self._get_indices(values)

    def _create_buffer(
        self, signal_probabilities: torch.Tensor, thetas: torch.Tensor, rhos: torch.Tensor, mode: str
    ) -> torch.Tensor:
        buffer = torch.zeros(signal_probabilities.shape[-2], signal_probabilities.shape[-1], dtype=torch.int)
        buffer = self._fill_buffer(buffer, thetas, rhos, mode)
        return buffer

    def _calculate_values(self, buffer: torch.Tensor, normalized_signal_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Calculate the sum of normalized_signal_probabilities in each bin of the buffer, where bins are defined by the buffer.

        Args:
            buffer (torch.Tensor): The buffer defining the bins (contains integers between 1 and self.granularity).
            normalized_signal_probabilities (torch.Tensor): The normalized signal probabilities.

        Returns:
            torch.Tensor: The values of the bins (sum of values is 1).
        """
        device = normalized_signal_probabilities.device
        # Match one_hot's range checks while transferring only one bin index per pixel.
        lowest, highest = torch.aminmax(buffer)
        if lowest < 0:
            raise RuntimeError("Class values must be non-negative.")
        if highest >= self.granularity:
            raise RuntimeError("Class values must be smaller than num_classes.")
        one_hot_buffer = buffer.to(device).unsqueeze(-1) == torch.arange(self.granularity, device=device)
        values = (one_hot_buffer * normalized_signal_probabilities.unsqueeze(-1)).sum((0, 1))
        return values

    def _initialize_parameters(
        self, params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rhos = torch.linspace(params["rho_max"], params["rho_min"], self.granularity)
        thetas_horizontal = torch.linspace(
            params["theta_min_horizontal"], params["theta_max_horizontal"], self.granularity
        )
        thetas_vertical = torch.linspace(params["theta_min_vertical"], params["theta_max_vertical"], self.granularity)
        return rhos, thetas_horizontal, thetas_vertical

    def _line_intersection_from_hough(
        self, rho_first: torch.Tensor, theta_first: torch.Tensor, rho_second: torch.Tensor, theta_second: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Solves for the intersection point `(x, y)` of two lines represented in Hough space:

        .. math::

            x \sin(\theta_{first}) + y \cos(\theta_{first}) = \rho_{first} \\
            x \sin(\theta_{second}) + y \cos(\theta_{second}) = \rho_{second}

        Args:
            rho_first (torch.Tensor): The radial distance of the first line.
            theta_first (torch.Tensor): The angle of the first line in radians.
            rho_second (torch.Tensor): The radial distance of the second line.
            theta_second (torch.Tensor): The angle of the second line in radians.

        Returns:
            torch.Tensor: The intersection point `(x, y)` of the two lines.
        """

        det = torch.cos(theta_first) * torch.sin(theta_second) - torch.cos(theta_second) * torch.sin(theta_first)
        if det.abs() < 1e-6:
            raise ValueError("Lines are parallel")
        x = (rho_first * torch.sin(theta_second) - rho_second * torch.sin(theta_first)) / det
        y = (rho_second * torch.cos(theta_first) - rho_first * torch.cos(theta_second)) / det
        return torch.stack((x, y))

    def _get_indices(self, values: torch.Tensor) -> Tuple[int, int]:
        cumsum_values = values.cumsum(0) / values.sum()
        lower_bound: int = int((cumsum_values >= self.percentiles[0]).nonzero().min())
        upper_bound: int = int((cumsum_values <= self.percentiles[1]).nonzero().max())
        width = (upper_bound - lower_bound) / (self.percentiles[1] - self.percentiles[0])
        lower_bound = int(lower_bound) - int(width * self.percentiles[0])
        upper_bound = int(upper_bound) + int(width * (1 - self.percentiles[1]))
        return lower_bound, upper_bound

    def _normalize_signal_probabilities(self, signal_probabilities: torch.Tensor) -> torch.Tensor:
        normalized_signal_probabilities = signal_probabilities / signal_probabilities.sum()
        return normalized_signal_probabilities.squeeze()

    def _fill_buffer(self, buffer: torch.Tensor, thetas: torch.Tensor, rhos: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "horizontal":
            y = torch.arange(buffer.shape[-2])
            for theta, rho in zip(thetas, rhos):
                x = self._y_to_x(y, theta, rho)
                x = x.round().long().clamp(min=0, max=buffer.shape[-1] - 1)
                buffer[y, x] += 1
            buffer = buffer.cumsum(dim=1)
        elif mode == "vertical":
            x = torch.arange(buffer.shape[-1])
            for theta, rho in zip(thetas, rhos):
                y = self._x_to_y(x, theta, rho)
                y = y.round().long().clamp(min=0, max=buffer.shape[-2] - 1)
                buffer[y, x] += 1
            buffer = buffer.cumsum(dim=0)
        else:
            raise ValueError("mode must be either 'horizontal' or 'vertical'")

        return self._fix_edge_effects(buffer)

    def _fix_edge_effects(self, buffer: torch.Tensor) -> torch.Tensor:
        buffer[0, :] = buffer[1, :]
        buffer[-1, :] = buffer[-2, :]
        buffer[:, 0] = buffer[:, 1]
        buffer[:, -1] = buffer[:, -2]
        return buffer

    def _x_to_y(self, x: torch.Tensor, theta: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        r"""
        Solves for y in

        .. math::

            x \sin(\theta) + y \cos(\theta) = \rho

        and evaluates at the given x-coordinates.

        Args:
            x (torch.Tensor): The x-coordinates.
            theta (torch.Tensor): The angle in radians.
            rho (torch.Tensor): The radial distance.

        Returns:
            torch.Tensor: The y-coordinates.
        """
        return -x / torch.tan(theta) + rho / torch.sin(theta)

    def _y_to_x(self, y: torch.Tensor, theta: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        r"""
        Solves for x in

        .. math::

            x \sin(\theta) + y \cos(\theta) = \rho

        and evaluates at the given y-coordinates.

        Args:
            y (torch.Tensor): The y-coordinates.
            theta (torch.Tensor): The angle in radians.
            rho (torch.Tensor): The radial distance.

        Returns:
            torch.Tensor: The x-coordinates.
        """
        return -y * torch.tan(theta) + rho / torch.cos(theta)

    def apply_perspective(
        self, input_tensor: torch.Tensor, source_points: torch.Tensor, fill_value: float | List[float] = 0
    ) -> torch.Tensor:
        original_shape = input_tensor.shape
        if input_tensor.ndim == 2:  # (H, W)
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.ndim == 3:  # (C, H, W)
            input_tensor = input_tensor.unsqueeze(0)

        H, W = input_tensor.shape[-2:]
        destination_points = self._calculate_destination_points(H, W, source_points)
        corrected: torch.Tensor = perspective(
            input_tensor, source_points.tolist(), destination_points.tolist(), fill=fill_value
        )

        if len(original_shape) == 4:
            return corrected
        elif len(original_shape) == 3:
            return corrected.squeeze(0)
        else:
            return corrected.squeeze(0).squeeze(0)

    def _get_approximate_aspect_ratio(self, points: torch.Tensor) -> float:
        r"""
        Returns the approximate aspect ratio of the quadrilateral defined by the source points.

        Args:
            source_points (torch.Tensor): The source points defining the quadrilateral.

        Returns:
            float: The approximate aspect ratio of the quadrilateral.
        """
        left = points[0, 0] + points[3, 0]
        right = points[1, 0] + points[2, 0]
        top = points[0, 1] + points[1, 1]
        bottom = points[2, 1] + points[3, 1]

        aspect = ((right - left) / (bottom - top)).float().item()

        return aspect

    def _calculate_destination_points(self, H: int, W: int, source_points: torch.Tensor) -> torch.Tensor:
        aspect_ratio_source = self._get_approximate_aspect_ratio(source_points)

        if W < aspect_ratio_source * H:
            target_H = W / aspect_ratio_source
            destination_points = [
                [0, (H - target_H) / 2],
                [W, (H - target_H) / 2],
                [W, (H + target_H) / 2],
                [0, (H + target_H) / 2],
            ]
        else:
            target_W = H * aspect_ratio_source
            destination_points = [
                [(W - target_W) / 2, 0],
                [(W + target_W) / 2, 0],
                [(W + target_W) / 2, H],
                [(W - target_W) / 2, H],
            ]

        default_destination_points = [
            [W / 2, H / 2],
            [W / 2, H / 2],
            [W / 2, H / 2],
            [W / 2, H / 2],
        ]

        return self.alpha * torch.tensor(destination_points) + (1 - self.alpha) * torch.tensor(
            default_destination_points
        )
