import os
from typing import Any, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
from skimage.feature import peak_local_max
from sklearn.neighbors import NearestNeighbors
from torch_tps import ThinPlateSpline

DEBUG = False
# If DEBUG is True, specify a directory to save plots
DEBUG_OUTPUT_DIR = "sandbox"
if DEBUG and not os.path.exists(DEBUG_OUTPUT_DIR):
    os.makedirs(DEBUG_OUTPUT_DIR)


class Dewarper(torch.nn.Module):
    def __init__(
        self,
        min_peak_distance_factor: float = 0.7,
        abs_peak_threshold: float = 0.0,
        direction_norm_threshold: float = 0.95,
        magnitude_threshold: float = 0.95,
        optimizer_lr: float = 1.0,
        optimizer_steps: int = 1000,
        optimizer_lr_decay_rate: float = 0.999,
        max_num_warp_points: int = 75,
    ) -> None:
        """
        Initializes the Dewarper with parameters. The dewarper builds a grid of points based on input feature map.
        The grid is then optimized to align with the expected layout of the input data, i.e. a regular grid.

        Args:
            min_peak_distance_factor (float): Factor for min_distance in peak_local_max.
            abs_peak_threshold (float): Absolute threshold for peak_local_max.
            direction_norm_threshold (float): Threshold for direction norm filtering of nodes.
            magnitude_threshold (float): Threshold for magnitude filtering of nodes.
            optimizer_lr (float): Learning rate for the layout optimization.
            optimizer_steps (int): Number of optimization steps for node layout.
            optimizer_lr_decay_rate (float): Decay rate for the learning rate.
            max_num_warp_points (int): Maximum number of control points for ThinPlateSpline fitting.
        """
        self.min_peak_distance_factor = min_peak_distance_factor
        self.abs_peak_threshold = abs_peak_threshold
        self.direction_norm_threshold = direction_norm_threshold
        self.magnitude_threshold = magnitude_threshold
        self.optimizer_lr = optimizer_lr
        self.optimizer_steps = optimizer_steps
        self.optimizer_lr_decay_rate = optimizer_lr_decay_rate
        self.max_num_warp_points = max_num_warp_points
        self.nn_neighbors = 5  # Number of neighbors for KNN (including self, so 5 means 4 neighbors)
        self.kernel_m = 4  # Grid has 4-fold symmetry

        # These will be set in __call__
        self.grid_probabilities: torch.Tensor
        self.pixels_per_mm: float
        self.device: torch.device
        self.target_grid_size: float
        self.kernel_size: int
        self.grid: torch.Tensor

        self.multidim_kernel: torch.Tensor
        self.channel: torch.Tensor
        self.local_maxima: npt.NDArray[Any]
        self.final_local_maxima: npt.NDArray[Any]
        self.final_edges: list[tuple[int, int]]
        self.optimized_positions: torch.Tensor

    def _spherical_harmonic_kernel(self, size: int) -> torch.Tensor:
        """
        Create a directional kernel using the real part of 2D spherical harmonics.

        Args:
            size (int): The size of the square kernel (e.g., 21 for a 21x21 kernel).
                        Must be an odd number for symmetric kernel.

        Returns:
            torch.Tensor: A multi-dimensional kernel tensor.
        """
        assert size % 2 == 1, "Size must be odd for symmetric kernel"
        half = size // 2

        # Use torch.meshgrid and torch operations
        y_coords, x_coords = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=self.device),
            torch.arange(-half, half + 1, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        r = torch.sqrt(x_coords**2 + y_coords**2) + 1e-6  # add epsilon to avoid div-by-zero
        phi = torch.atan2(y_coords, x_coords)
        basis_fcn = torch.cos(self.kernel_m * phi)  # real spherical harmonic: directional pattern
        basis_fcn *= torch.exp(-(r**2) / (2 * (half / 2) ** 2))  # optional Gaussian envelope

        kernel = basis_fcn

        thetas = [0]  # angles in degrees
        zooms = [1]  # zoom factors
        num_rolls = len(thetas)
        num_zooms = len(zooms)

        xc = torch.linspace(-1, 1, kernel.shape[0], device=self.device)
        x_grid, y_grid = torch.meshgrid(xc, xc, indexing="ij")
        coordinates = torch.stack((x_grid, y_grid), dim=-1).reshape(-1, 2)

        def get_transformation(theta: float, zoom: float) -> torch.Tensor:
            """
            Get 2D transformation matrix for angle theta in degrees and zoom factor.

            Args:
                theta (float): Rotation angle in degrees.
                zoom (float): Zoom factor.

            Returns:
                torch.Tensor: The 2D transformation matrix.
            """
            theta_rad = np.deg2rad(theta)  # np.deg2rad is fine as it's a scalar op
            return (
                torch.tensor(
                    [[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]],
                    dtype=torch.float32,
                    device=self.device,
                )
                * zoom
            )

        multidim_ctr = 0
        multidim_kernel = torch.zeros((num_rolls * num_zooms, kernel.shape[0], kernel.shape[0]), device=self.device)
        for theta in thetas:
            for zoom in zooms:
                transformation = get_transformation(theta, zoom)
                transformed_coordinates = coordinates @ transformation.T
                intermediate = torch.nn.functional.grid_sample(
                    kernel.unsqueeze(0).unsqueeze(0),  # kernel is already a torch.Tensor
                    transformed_coordinates.reshape(1, kernel.shape[0], kernel.shape[0], 2),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze()
                intermediate /= intermediate.max()
                multidim_kernel[multidim_ctr] = intermediate
                multidim_ctr += 1

        return multidim_kernel

    def _perform_convolution(self) -> None:
        """
        Performs 2D convolution on the grid probabilities with the generated kernel
        and finds local maxima.
        """
        self.target_grid_size = 5 * self.pixels_per_mm
        self.kernel_size = int(10 * self.pixels_per_mm)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.multidim_kernel = self._spherical_harmonic_kernel(self.kernel_size)

        grid_probs_unsqueezed = self.grid_probabilities.unsqueeze(0).unsqueeze(0).to(self.device)
        k = self.multidim_kernel.unsqueeze(1).float().to(self.device)

        grid_probabilities_conv = torch.nn.functional.conv2d(
            grid_probs_unsqueezed, k, padding=self.multidim_kernel.shape[-1] // 2
        )
        self.channel = grid_probabilities_conv.sum(1).squeeze().cpu()  # Move to CPU for peak_local_max
        self.channel /= self.channel.max()

        self.local_maxima = peak_local_max(
            self.channel.numpy(),  # peak_local_max requires numpy
            min_distance=int(self.target_grid_size * self.min_peak_distance_factor),
            threshold_abs=self.abs_peak_threshold,
        )  # type: ignore

        if DEBUG:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.channel.numpy(), cmap="gray")
            plt.scatter(
                self.local_maxima[:, 1],
                self.local_maxima[:, 0],
                c="red",
                s=9,
                label="Local Maxima",
            )
            plt.title("Grid Probabilities after Convolution and Local Maxima")
            plt.colorbar()
            plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, "convolution_maxima.png"))
            plt.show()

    def _filter_and_graph_nodes(self) -> None:
        """
        Filters local maxima based on directionality and magnitude, then constructs
        a graph, keeping only the largest connected component.
        """
        if self.local_maxima.shape[0] == 0:
            self.final_local_maxima = np.array([])
            self.final_edges = []
            return

        knn = NearestNeighbors(n_neighbors=self.nn_neighbors, algorithm="ball_tree", p=1)
        knn.fit(self.local_maxima)
        _, indices = knn.kneighbors(self.local_maxima)

        direction_norms = []
        magnitudes = []
        for i in range(len(self.local_maxima)):
            center = self.local_maxima[i]
            vectors = self.local_maxima[indices[i][1:]] - center

            # Convert to torch tensor for magnitude calculation
            vectors_t = torch.tensor(vectors, dtype=torch.float32, device=self.device)
            vector_sum_norm = torch.norm(torch.sum(vectors_t, dim=0))
            mean_abs_vector_norm = torch.norm(torch.mean(torch.abs(vectors_t), dim=0))

            magnitude = (
                vector_sum_norm / mean_abs_vector_norm
                if mean_abs_vector_norm != 0
                else torch.tensor(0.0, device=self.device)
            )
            magnitudes.append(magnitude.item())  # Store as float

            # Normalize vectors to calculate cosine similarity
            norm_vectors = vectors_t / torch.norm(vectors_t, dim=1, keepdim=True)
            cos_sim = torch.flatten(norm_vectors @ norm_vectors.T)

            sorted_cos_sim = torch.sort(cos_sim)[0]
            cos_val = torch.prod(sorted_cos_sim[:4:2]).item()  # Store as float

            direction_norms.append(cos_val)

        direction_norms_arr = np.array(direction_norms)
        magnitudes_arr = np.array(magnitudes)

        keep_mask = (direction_norms_arr >= self.direction_norm_threshold) * (magnitudes_arr < self.magnitude_threshold)
        refined_local_maxima = self.local_maxima[keep_mask]

        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(keep_mask)[0])}
        G: nx.Graph[int] = nx.Graph()

        for i, original_idx in enumerate(np.where(keep_mask)[0]):
            for j in indices[original_idx][1:]:  # Skip self
                if keep_mask[j]:
                    G.add_edge(i, idx_map[j])

        if len(G) > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        else:
            self.final_local_maxima = np.array([])
            self.final_edges = []
            return

        final_indices = sorted(G.nodes)
        self.final_local_maxima = refined_local_maxima[final_indices]
        index_remap = {old: new for new, old in enumerate(final_indices)}
        self.final_edges = [(index_remap[u], index_remap[v]) for u, v in G.edges]

        if DEBUG:
            plt.figure(figsize=(10, 10))
            plt.imshow(self.channel.numpy(), cmap="gray")
            for src, dst in self.final_edges:
                plt.plot(
                    [self.final_local_maxima[src, 1], self.final_local_maxima[dst, 1]],
                    [self.final_local_maxima[src, 0], self.final_local_maxima[dst, 0]],
                    c="blue",
                    alpha=0.8,
                )
            plt.scatter(
                self.final_local_maxima[:, 1],
                self.final_local_maxima[:, 0],
                c="green",
                s=15,
            )
            plt.title("Largest Connected Component of Filtered KNN Graph")
            plt.colorbar()
            plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, "filtered_graph.png"))
            plt.show()

    def _calculate_distances(self, positions: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Calculate distances between nodes based on their positions and edges.

        Args:
            positions (torch.Tensor): [N, 2] tensor of node positions (float).
            edges (torch.Tensor): [E, 2] tensor of edge indices (long).

        Returns:
            torch.Tensor: [E] tensor of distances between connected nodes.
        """
        p1 = positions[edges[:, 0]]
        p2 = positions[edges[:, 1]]
        distances: torch.Tensor = torch.norm(p1 - p2, dim=1)
        return distances

    def _layout_loss(self, positions: torch.Tensor, edges: torch.Tensor, target_distance: float) -> torch.Tensor:
        """
        Calculates the layout loss based on deviations from the target distance.

        Args:
            positions (torch.Tensor): [N, 2] tensor of node positions (float).
            edges (torch.Tensor): [E, 2] tensor of edge indices (long).
            target_distance (float): The desired distance between connected nodes.

        Returns:
            torch.Tensor: The calculated layout loss.
        """
        p1 = positions[edges[:, 0]]
        p2 = positions[edges[:, 1]]
        diff = (p1 - p2).abs()
        max_diff = diff.max(dim=1).values
        min_diff = diff.min(dim=1).values
        loss = ((max_diff - target_distance).pow(2) + min_diff.pow(2)).mean().sqrt()
        return loss

    def _plot_positions(self, pos: torch.Tensor, title: str, filename: str, c: npt.NDArray[Any] | None = None) -> None:
        """
        Plots node positions.

        Args:
            pos (torch.Tensor): Tensor of node positions.
            title (str): Title of the plot.
            filename (str): Filename to save the plot.
            c (np.ndarray | None): Color values for scatter plot, if any.
        """
        pos_cpu = pos.detach().cpu().numpy()
        plt.figure(figsize=(10, 5))
        if c is not None:
            plt.scatter(pos_cpu[:, 1], pos_cpu[:, 0], s=10, c=c, alpha=0.5, cmap="jet")
        else:
            plt.scatter(pos_cpu[:, 1], pos_cpu[:, 0], s=10, color="blue", alpha=0.5)
        plt.title(title)
        plt.xlabel("Column Index")
        plt.ylabel("Vertical Position")
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, filename))
        plt.show()

    def _get_node_comfort(
        self,
        positions: torch.Tensor,
        edges_tensor: torch.Tensor,
        mean_dist: float,
    ) -> list[float]:
        """
        Calculates a 'comfort' metric for each node based on neighbor distances.

        Args:
            positions (torch.Tensor): Tensor of node positions.
            edges_tensor (torch.Tensor): Tensor of edges.
            mean_dist (float): The target mean distance between nodes.

        Returns:
            list[float]: A list of comfort values for each node.
        """
        node_comfort = []
        for node_idx in range(positions.shape[0]):
            # Find edges where the current node is the source
            outgoing_edges_mask = edges_tensor[:, 0] == node_idx
            # Find edges where the current node is the destination
            incoming_edges_mask = edges_tensor[:, 1] == node_idx

            # Combine neighbors from both directions
            all_neighbors_indices = torch.cat(
                (edges_tensor[outgoing_edges_mask, 1], edges_tensor[incoming_edges_mask, 0])
            ).unique()  # type: ignore

            if len(all_neighbors_indices) > 0:
                neighbor_positions = positions[all_neighbors_indices]
                distances = torch.norm(neighbor_positions - positions[node_idx], dim=1)
                comfort = (distances - mean_dist).abs().mean().item()
            else:
                comfort = 0.0  # Node with no connections
            node_comfort.append(comfort)
        return node_comfort

    def _decay_lr(self, optimizer: torch.optim.Adam, decay_rate: float) -> None:
        """
        Decays the learning rate of the optimizer.

        Args:
            optimizer (torch.optim.Adam): The optimizer whose learning rate will be decayed.
            decay_rate (float): The decay rate.
        """
        for param_group in optimizer.param_groups:
            param_group["lr"] *= decay_rate

    def _optimize_grid_layout(self) -> None:
        """
        Optimizes the layout of the grid points using gradient descent.
        """
        if self.final_local_maxima.shape[0] == 0:
            self.optimized_positions = torch.tensor([])
            return

        coordinates = torch.from_numpy(self.final_local_maxima.copy()).float().to(self.device)
        edges_tensor = torch.tensor(self.final_edges, dtype=torch.long).to(self.device)

        with cast(Any, torch.enable_grad)():
            positions = torch.nn.Parameter(coordinates, requires_grad=True).to(self.device)
            optimizer = torch.optim.Adam([positions], lr=self.optimizer_lr)

            if DEBUG:
                self._plot_positions(positions, "Initial Positions of Nodes", "initial_positions.png")

            for step in range(self.optimizer_steps):
                optimizer.zero_grad()
                loss = self._layout_loss(positions, edges_tensor, target_distance=self.target_grid_size)
                loss.backward()  # type: ignore
                optimizer.step()
                self._decay_lr(optimizer, decay_rate=self.optimizer_lr_decay_rate)

            self.optimized_positions = positions.detach().cpu()

        if DEBUG:
            node_comfort = self._get_node_comfort(positions, edges_tensor, self.target_grid_size)
            self._plot_positions(
                positions,
                "Final Positions of Nodes after Optimization",
                "final_optimized_positions.png",
                c=np.array(node_comfort),
            )

    def _fit_warp(self, device: torch.device = torch.device("cpu")) -> None:
        """
        Warps the original grid probabilities image to the optimized grid layout
        using Thin Plate Spline (TPS).

        Returns:
            torch.Tensor: The warped image.
        """

        input_ctrl = torch.tensor(self.final_local_maxima, dtype=torch.float32)
        output_ctrl = self.optimized_positions

        height, width = self.grid_probabilities.shape
        size = torch.tensor([height, width], dtype=torch.float32).to(device)

        tps = ThinPlateSpline(1, device=device, order=1)

        # Sample control points for TPS if there are too many
        indices = torch.randperm(input_ctrl.shape[0])[: self.max_num_warp_points]
        sampled_input_ctrl = input_ctrl[indices]
        sampled_output_ctrl = output_ctrl[indices]
        try:
            tps.fit(sampled_output_ctrl, sampled_input_ctrl)
        except Exception as e:
            print(f"Error fitting TPS: {e}. The dewarping failed.")
            tps.fit(
                torch.tensor([[1.0, 2.0], [1.0, 3.0]]), torch.tensor([[1.0, 2.0], [1.0, 3.0]])
            )  # Fallback to identity transform

        i = torch.arange(height, dtype=torch.float32)
        j = torch.arange(width, dtype=torch.float32)

        ii, jj = torch.meshgrid(i, j, indexing="ij")
        output_indices = torch.stack((ii, jj), dim=-1).reshape(-1, 2)
        if self.final_local_maxima.shape[0] == 0 or self.optimized_positions.shape[0] == 0:

            self.grid = output_indices.reshape(height, width, 2).to(device)

        input_indices = tps.transform(output_indices).reshape(height, width, 2).to(device)

        grid = 2 * input_indices / size - 1
        self.grid = torch.flip(grid, (-1,)).to(device)

    def transform(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.grid is None:
            raise ValueError("Grid has not been initialized. Call fit() first.")
        warped = torch.nn.functional.grid_sample(
            feature_map.unsqueeze(0).unsqueeze(0), self.grid[None, ...].to(feature_map.device), align_corners=False
        )[0]
        if DEBUG:
            plt.figure(figsize=(10, 5))
            plt.imshow(feature_map.cpu().numpy(), cmap="gray")
            plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, "original_image.png"))
            plt.show()

            plt.figure(figsize=(10, 5))
            plt.imshow(warped.permute(1, 2, 0).cpu().squeeze().numpy(), cmap="gray")
            plt.savefig(os.path.join(DEBUG_OUTPUT_DIR, "warped_image.png"))
            plt.show()

        return warped.squeeze()

    def fit(self, grid_probabilities: torch.Tensor, pixels_per_mm: float) -> None:
        """
        Executes the full grid processing pipeline for a given image.

        Args:
            grid_probabilities (torch.Tensor): The input grid probabilities tensor of shape (H, W).
            pixels_per_mm (float): Pixels per millimeter, used for kernel and target grid size calculations.

        Returns:
            torch.Tensor: The final warped grid probabilities image on CPU.
        """
        self.grid_probabilities = grid_probabilities
        self.pixels_per_mm = pixels_per_mm
        self.device = grid_probabilities.device

        self._perform_convolution()
        self._filter_and_graph_nodes()
        self._optimize_grid_layout()
        self._fit_warp()
