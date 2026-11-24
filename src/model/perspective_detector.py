from typing import Dict, Tuple

import torch

from src.seed import image_generator

DEBUG = False

if DEBUG:
    import matplotlib.pyplot as plt


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    return torch.mean(image, dim=0, keepdim=True)


class PerspectiveDetector:

    def __init__(self, num_thetas: int, max_num_nonzero: int = 100_000) -> None:
        self.num_thetas = num_thetas
        self.max_num_nonzero = max_num_nonzero

    def hough_transform(self, image: torch.Tensor, thetas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Hough Transform to detect lines in an image.

        Args:
            image (torch.Tensor): The input image tensor.
            thetas (torch.Tensor): The angles (thetas) for the Hough transform, in radians.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The accumulator array, the rhos (radial distances).
        """
        device = image.device
        H, W = image.shape
        diag_len = int(torch.sqrt(torch.tensor(H**2 + W**2, device=device)))
        rhos = torch.linspace(-diag_len, diag_len, 2 * diag_len, device=device)
        num_thetas = len(thetas)
        num_rhos = len(rhos)

        y_idxs, x_idxs = torch.nonzero(image, as_tuple=True)
        if len(y_idxs) > self.max_num_nonzero:
            indices = torch.randperm(len(y_idxs), generator=image_generator())[: self.max_num_nonzero]
            y_idxs = y_idxs[indices]
            x_idxs = x_idxs[indices]
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)

        x_idxs = x_idxs.view(-1, 1).float()
        y_idxs = y_idxs.view(-1, 1).float()
        rhos_vals = x_idxs * cos_thetas + y_idxs * sin_thetas
        rhos_idxs = torch.round((rhos_vals - rhos[0]) / (rhos[1] - rhos[0])).int()
        rhos_idxs = rhos_idxs.clamp(0, len(rhos) - 1)

        accumulator = torch.zeros(num_rhos * num_thetas, dtype=torch.int32, device=device)
        idxs_flat = rhos_idxs * num_thetas + torch.arange(num_thetas, device=device).reshape(1, -1)

        idxs_flat = idxs_flat.flatten()

        accumulator.index_add_(0, idxs_flat, torch.ones_like(idxs_flat, dtype=torch.int32))
        accumulator = accumulator.view(len(rhos), len(thetas))

        # Remove most of the negative rhos
        rhos = rhos[3 * num_rhos // 7 :]
        accumulator = accumulator[3 * num_rhos // 7 :]

        # The edges of the accumulator are set to the values of the adjacent cells
        # to avoid edge effects in the line extraction
        accumulator[:, 0] = accumulator[:, 1]
        accumulator[:, -1] = accumulator[:, -2]
        accumulator[:, num_thetas // 2 - 1] = accumulator[:, num_thetas // 2 - 2]
        accumulator[:, num_thetas // 2] = accumulator[:, num_thetas // 2 + 1]

        return accumulator.float(), rhos

    def get_line_values(
        self, accumulator: torch.Tensor, thetas: torch.Tensor, rhos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Extract the values along a line in the Hough accumulator, parameterized by two thetas: one at the top and
        one at the bottom. The thetas are chosen to maximize the variance along the line and correspond to the
        angles of the lines in the image.

        Args:
            accumulator (torch.Tensor): The Hough accumulator tensor.
            thetas (torch.Tensor): The angles (thetas) corresponding to dim 1 of the tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The values along the line, the corresponding thetas,
                and the radial distances.
        """
        variance = self.calculate_line_variances(accumulator)
        max_idx_top = variance.argmax() // (self.num_thetas // 2)
        max_idx_bottom = variance.argmax() % (self.num_thetas // 2)

        return max_idx_top, max_idx_bottom

    def get_theta_lims(self, accumulator: torch.Tensor, thetas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the variance along lines in the Hough accumulator, parameterized by two thetas: one at the top and
        one at the bottom. The thetas are chosen to maximize the variance along the line and correspond to the
        angles of the lines in the image.

        Args:
            accumulator (torch.Tensor): The Hough accumulator tensor.
            thetas (torch.Tensor): The angles (thetas) corresponding to dim 1 of the tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The thetas corresponding to the top and bottom of the accumulator.
        """
        variance = self.calculate_line_variances(accumulator)
        max_idx_top = variance.argmax() // (self.num_thetas // 2)
        max_idx_bottom = variance.argmax() % (self.num_thetas // 2)

        theta_top = thetas[max_idx_top]  # index at top of accumulator
        theta_bottom = thetas[max_idx_bottom]  # index at bottom of accumulator

        return theta_top, theta_bottom

    def find_line_params(self, image: torch.Tensor, eps: float) -> Dict[str, torch.Tensor]:
        r"""
        Find the defining parameters of the two lines in hough domain corresponding to the vertical and horizontal gridlines in image domain.

        Args:
            image (torch.Tensor): The input image tensor, with shape [H, W]. Should be a binary image.
            eps (float): Margin added to the thetas for the second pass of the Hough transform.

        Returns:
            Dict[str, torch.Tensor]: The defining parameters of the two lines in hough domain.
        """
        hann = (
            torch.cat([torch.hann_window(self.num_thetas // 2), torch.hann_window(self.num_thetas // 2)])
            .unsqueeze(0)
            .to(image.device)
        ).pow(0.25)

        # PASS 1
        thetas = self.get_initial_thetas().to(image.device)
        accumulator, rhos = self.hough_transform(image, thetas)
        accumulator = accumulator * hann

        if DEBUG:
            fig, ax = plt.subplots(1, 1, figsize=(25, 15))
            ax.imshow(accumulator.cpu().numpy(), aspect="auto", cmap="nipy_spectral", interpolation="nearest")
            plt.tight_layout()
            plt.savefig("sandbox/accumulator1.png")
            plt.show()

        (
            theta_top_horizontal,
            theta_bottom_horizontal,
        ) = self.get_theta_lims(accumulator[:, : self.num_thetas // 2], thetas[: self.num_thetas // 2])
        (
            theta_top_vertical,
            theta_bottom_vertical,
        ) = self.get_theta_lims(accumulator[:, self.num_thetas // 2 :], thetas[self.num_thetas // 2 :])

        theta_min_horizontal = torch.min(theta_top_horizontal, theta_bottom_horizontal)
        theta_max_horizontal = torch.max(theta_top_horizontal, theta_bottom_horizontal)
        theta_min_vertical = torch.min(theta_top_vertical, theta_bottom_vertical)
        theta_max_vertical = torch.max(theta_top_vertical, theta_bottom_vertical)

        # PASS 2
        thetas = torch.cat(
            [
                torch.linspace(
                    theta_min_horizontal - eps,
                    theta_max_horizontal + eps,
                    self.num_thetas // 2,
                    device=image.device,
                ),
                torch.linspace(
                    theta_min_vertical - eps,
                    theta_max_vertical + eps,
                    self.num_thetas // 2,
                    device=image.device,
                ),
            ]
        )
        accumulator, rhos = self.hough_transform(image, thetas)
        accumulator = accumulator * hann

        if DEBUG:
            fig, ax = plt.subplots(1, 1, figsize=(25, 15))
            ax.imshow(accumulator.cpu().numpy(), aspect="auto", cmap="nipy_spectral", interpolation="nearest")
            plt.tight_layout()
            plt.savefig("sandbox/accumulator2.png")
            plt.show()

        idx_top_horizontal, idx_bottom_horizontal = self.get_line_values(
            accumulator[:, : self.num_thetas // 2], thetas[: self.num_thetas // 2], rhos
        )
        idx_top_vertical, idx_bottom_vertical = self.get_line_values(
            accumulator[:, self.num_thetas // 2 :], thetas[self.num_thetas // 2 :], rhos
        )

        params = {
            "rho_min": rhos[0],
            "rho_max": rhos[-1],
            "theta_min_vertical": thetas[self.num_thetas // 2 + idx_bottom_vertical],
            "theta_max_vertical": thetas[self.num_thetas // 2 + idx_top_vertical],
            "theta_min_horizontal": thetas[idx_bottom_horizontal],
            "theta_max_horizontal": thetas[idx_top_horizontal],
        }
        return params

    def get_initial_thetas(self) -> torch.Tensor:
        return torch.cat(
            [
                torch.linspace(-torch.pi / 4, torch.pi / 4, self.num_thetas // 2),
                torch.linspace(torch.pi / 4, 3 * torch.pi / 4, self.num_thetas // 2),
            ]
        )

    def __call__(self, image: torch.Tensor, eps: float = torch.pi / 180) -> Dict[str, torch.Tensor]:
        r"""
        Correct the perspective of an image of a gridded paper.

        Args:
            image (torch.Tensor): The input image tensor, with shape [B, C, H, W], [C, H, W] or [H,W].
            eps (float): Margin added to the thetas for the second pass of the Hough transform.

        Returns:
            Dict[str, torch.Tensor]: The defining parameters of the two lines in hough domain.

        """
        binary_image = self.binarize(image)

        if DEBUG:
            fig, ax = plt.subplots(1, 1, figsize=(25, 15))
            ax.imshow(binary_image.cpu().numpy(), cmap="gray")
            plt.tight_layout()
            plt.savefig("sandbox/binary_image.png")
            plt.show()

        params = self.find_line_params(binary_image, eps)
        return params

    def quantile(self, tensor: torch.Tensor, q: float, max_num_elems: int = 10_000) -> torch.Tensor:
        r"""
        Compute the q-th quantile of a tensor.

        Args:
            tensor (torch.Tensor): The input tensor.
            q (float): The quantile to compute.

        Returns:
            torch.Tensor: The q-th quantile of the tensor.
        """
        flattened_tensor = tensor.flatten()
        if len(flattened_tensor) > max_num_elems:
            indices = torch.randperm(len(flattened_tensor), generator=image_generator())[:max_num_elems]
            flattened_tensor = flattened_tensor[indices]
        return torch.quantile(flattened_tensor, q)

    def to_flat_greyscale(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.ndim == 4:
            image = rgb_to_grayscale(image.squeeze(0)).squeeze(0)
        elif image.ndim == 3:
            image = rgb_to_grayscale(image).squeeze(0)
        return (image - image.min()) / (image.max() - image.min())

    def binarize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): The input image tensor, with shape [C, H, W] or [H, W].

        Returns:
            torch.Tensor: The binarized image tensor with shape [H, W].
        """
        return image.squeeze() > self.quantile(image, 0.98)

    def calculate_line_variances(self, accumulator: torch.Tensor) -> torch.Tensor:
        """
        Calculate variance along the derivative for all lines from top to bottom of an image.

        Quadratic complexity in the number of thetas (W).

        Args:
            accumulator (torch.Tensor): The Hough accumulator tensor.

        Returns:
            torch.Tensor: The variance tensor with shape [W, W].
        """
        H, W = accumulator.shape
        x_start, x_end = torch.meshgrid(torch.arange(W), torch.arange(W), indexing="ij")
        slopes = torch.where(x_start != x_end, H / (x_end - x_start).float(), torch.tensor(float("inf")))
        # Keep the small slope calculation on the CPU. Sample and reduce where the accumulator lives.
        slopes = slopes.to(accumulator.device)
        x_start = x_start.to(accumulator.device)
        y_coords = torch.arange(H, device=accumulator.device).view(-1, 1, 1).float()
        x_coords = torch.where(
            slopes == float("inf"), x_start.float(), x_start + y_coords / slopes  # Diagonal entries: x_start == x_end
        )
        x_coords_clamped = x_coords.round_().clamp_(0, W - 1).long()
        del x_coords
        sampled_values = accumulator[y_coords.long(), x_coords_clamped]
        variances = torch.var(sampled_values, dim=0)
        return variances
