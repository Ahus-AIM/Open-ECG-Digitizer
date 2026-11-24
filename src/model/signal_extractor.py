from collections import defaultdict, deque
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torchvision
from scipy.ndimage import find_objects
from scipy.optimize import linear_sum_assignment
from skimage.measure import label


class SignalExtractor:
    threshold_sum: float
    threshold_line_in_mask: float
    label_thresh: float
    max_iterations: int
    split_num_stripes: int
    candidate_span: int
    debug: int
    lam: float
    min_line_width: int
    num_peaks: Optional[int]

    def __init__(
        self,
        threshold_sum: float = 10.0,
        threshold_line_in_mask: float = 0.95,
        label_thresh: float = 0.1,
        max_iterations: int = 4,
        split_num_stripes: int = 4,
        candidate_span: int = 10,
        debug: int = 0,
        lam: float = 0.5,
        min_line_width: int = 30,
    ) -> None:
        self.threshold_sum = threshold_sum
        self.threshold_line_in_mask = threshold_line_in_mask
        self.label_thresh = label_thresh
        self.max_iterations = max_iterations
        self.split_num_stripes = split_num_stripes
        self.candidate_span = candidate_span
        self.debug = debug
        self.lam = lam
        self.min_line_width = min_line_width
        self.num_peaks = None

    def __call__(self, feature_map: torch.Tensor) -> torch.Tensor:
        fmap = feature_map.cpu().clone()
        lines_list = self._iterative_extraction(fmap)
        self.num_peaks = self._autodetect_num_peaks(fmap)
        lines_list = [ln for ln in lines_list if (~torch.isnan(ln)).sum() > self.min_line_width]
        if len(lines_list) == 0:
            return torch.empty((0, feature_map.shape[1]), dtype=torch.float32)
        lines = torch.stack(lines_list, dim=0)
        merged_lines_list, overlaps = self.match_and_merge_lines(lines)
        if len(merged_lines_list) == 0:
            return torch.empty((0, feature_map.shape[1]), dtype=torch.float32)
        merged_lines = torch.stack(merged_lines_list, dim=0)
        if self.num_peaks != len(merged_lines):
            print(
                f"Warning: Number of peaks ({self.num_peaks}) does not match number of merged lines ({len(merged_lines)})."
            )
        return merged_lines

    def _iterative_extraction(self, fmap: torch.Tensor) -> list[torch.Tensor]:
        for it in range(self.max_iterations):
            good, rejected, rej_maps = self._extract_candidate_lines(fmap)
            if not rejected:
                break
            self._refine_fmap_by_removal(fmap, rej_maps)
        return good

    def _extract_candidate_lines(
        self, fmap: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        lab: npt.NDArray[Any] = self._label_regions(fmap)
        zero_mask = lab == 0
        big_offset = lab.max() + 1000  # Just some big number to avoid overlap

        for i in range(self.split_num_stripes):
            sl = slice(
                i * lab.shape[1] // self.split_num_stripes,
                (i + 1) * lab.shape[1] // self.split_num_stripes,
            )
            relab: npt.NDArray[np.int64] = label(lab[:, sl] > 0, connectivity=1)  # type: ignore
            lab[:, sl] += big_offset * i + relab
        lab[zero_mask] = 0

        good: list[torch.Tensor] = []
        rejected: list[torch.Tensor] = []
        rej_maps: list[torch.Tensor] = []
        # A column reduction is independent of all other columns. Pack disjoint
        # (label, column) pairs together while preserving the full reduction
        # shape, dtype and row order of the original masked sums.
        height, width = lab.shape
        ys, xs = np.nonzero(lab)
        if not xs.size:
            return good, rejected, rej_maps
        stride = int(lab.max()) + 1
        pairs, inverse = np.unique(xs * stride + lab[ys, xs], return_inverse=True)
        cols, labels = np.divmod(pairs, stride)
        starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
        slots = np.arange(len(cols)) - np.repeat(starts, np.diff(starts, append=len(cols)))
        pixel_slots = slots[inverse]
        centers = torch.empty(len(cols), dtype=fmap.dtype)
        for slot in range(int(slots.max()) + 1):
            pixels = pixel_slots == slot
            mask = torch.zeros((height, width), dtype=torch.bool)
            mask[ys[pixels], xs[pixels]] = True
            line = self._extract_line_from_region(fmap, mask)
            selected = slots == slot
            centers[selected] = line[cols[selected]]

        for lid in np.unique(labels):
            selected = labels == lid
            line = torch.full((width,), float("nan"), dtype=fmap.dtype)
            active_cols = cols[selected]
            values = centers[selected]
            line[active_cols] = values
            rows = values.long().clamp(0, height - 1).numpy()
            in_mask = torch.from_numpy(lab[rows, active_cols] == lid)
            if in_mask.float().mean().item() >= self.threshold_line_in_mask:
                good.append(line)
            else:
                rejected.append(line)
                rej_maps.append(fmap * torch.from_numpy(lab == lid))

        for line in (*good, *rejected):
            line[line < 5] = float("nan")

        return good, rejected, rej_maps

    def _refine_fmap_by_removal(self, fmap: torch.Tensor, rej_maps: list[torch.Tensor]) -> None:
        for i, mask_map in enumerate(rej_maps):
            ys, xs = torch.where(mask_map > 0)
            if ys.numel() == 0:
                continue
            y0, y1 = int(ys.min().item()), int(ys.max().item())
            x0, x1 = int(xs.min().item()), int(xs.max().item())
            cropped: npt.NDArray[Any] = mask_map[y0 : y1 + 1, x0 : x1 + 1].cpu().numpy()
            _, new_img = self._trace_horizontal_path(cropped)
            region = fmap[y0 : y1 + 1, x0 : x1 + 1]
            region[cropped > 0] = new_img[cropped > 0]

    def _label_regions(self, fmap: torch.Tensor) -> npt.NDArray[np.int64]:
        lab: npt.NDArray[np.int64] = label((fmap.numpy() > self.label_thresh).astype(int), connectivity=1)  # type: ignore
        for lb, bounds in enumerate(find_objects(lab), start=1):
            region = lab[bounds]
            mask = region == lb
            if fmap[bounds][mask].sum() < self.threshold_sum:
                region[mask] = 0
        output: npt.NDArray[np.int64] = label(lab > 0, connectivity=1)  # type: ignore
        return output

    def _extract_line_from_region(self, fmap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(fmap.shape[0]).view(-1, 1)
        masked = torch.where(mask, fmap, torch.zeros_like(fmap))
        masked /= masked.sum(0, keepdim=True).clamp(min=1e-6)
        line = (masked * pos).sum(0)
        line[line < 1 / fmap.shape[0]] = float("nan")
        return line

    def _trace_horizontal_path(self, img_arr: npt.NDArray[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        blurry = torch.tensor(img_arr, dtype=torch.float32)
        blurry += (torch.linspace(-1, 1, blurry.shape[0]) ** 6).unsqueeze(1)
        img = torch.tensor(img_arr, dtype=torch.float32)
        H, W = img.shape
        path_y: list[int] = [H // 2]

        candidate_count = 2 * self.candidate_span + 1
        penalty = torch.tensor(np.linspace(-1, 1, candidate_count) ** 2, dtype=torch.float32)
        # The walk depends on its previous row. Precompute only its deterministic
        # segment means, in bounded column blocks, and draw the jitter in walk order.
        rows = torch.arange(-self.candidate_span, H + self.candidate_span).clamp(0, H - 1)
        block_width = max(1, min(128, 65536 // (H * candidate_count)))
        for start in range(1, W, block_width):
            stop = min(start + block_width, W)
            columns = blurry[rows, start - 1 : stop].T.contiguous()
            scores = torch.empty((stop - start, H, candidate_count), dtype=blurry.dtype)
            for i, offset in enumerate(range(-self.candidate_span, self.candidate_span + 1)):
                first = self.candidate_span + min(offset, 0)
                windows = columns.unfold(1, abs(offset) + 1, 1)[:, first : first + H]
                means = windows.contiguous().mean(dim=-1)
                scores[:, :, i] = (means[:-1] + means[1:]) / 2

            for step in scores:
                prev = path_y[-1]
                vals = step[prev] + torch.randn(candidate_count) * 1e-6
                vals += vals.std() * penalty
                path_y.append(max(0, min(H - 1, prev + int(vals.argmin()) - self.candidate_span)))

        for i in range(len(path_y) - 1):
            lo, hi = sorted((path_y[i], path_y[i + 1]))
            img[lo - 1 : hi + 2, i] = 0

        return torch.tensor(path_y), img

    def _blur(self, tensor: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
        """
        Applies a Gaussian blur to the input tensor.

        Args:
            tensor (torch.Tensor): The input tensor to be blurred.
            kernel_size (int): The size of the Gaussian kernel.

        Returns:
            torch.Tensor: The blurred tensor.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = torchvision.transforms.GaussianBlur(kernel_size, sigma=sigma)
        blurred_tensor: torch.Tensor = kernel(tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        return blurred_tensor

    def _autodetect_num_peaks(self, fmap: torch.Tensor) -> int:
        logits = torch.log(1 - fmap.clamp(1e-6, 1 - 1e-6)) / torch.log(fmap.clamp(1e-6, 1 - 1e-6))
        logits = self._blur(logits)
        probs_raw = torch.softmax(logits, dim=0)
        probs = (probs_raw - 1.1 / probs_raw.shape[0]).clamp(0)

        zero_mask = probs == 0
        nonzero_mask = ~zero_mask
        lines = zero_mask[1:] & nonzero_mask[:-1]
        line_counts = lines.sum(0)
        value_counts = torch.bincount(line_counts)
        most_common_value = int(value_counts.argmax().item())
        most_common_count = int(value_counts.max().item())

        if most_common_count < line_counts.shape[0] * 0.5:
            print(
                f"Warning: Autodetected number of peaks ({most_common_value}) is not reliable, "
                f"consider setting it manually or adjusting the autodetection parameters."
            )

        return most_common_value

    def preprocess_lines(self, lines: torch.Tensor) -> torch.Tensor:
        lines = lines.clone()
        lines[lines == 0] = float("nan")
        valid_cols = lines.nan_to_num(0.0).abs().sum(0) > 0
        first, last = torch.nonzero(valid_cols, as_tuple=True)[0][[0, -1]].tolist()
        return lines[:, first : last + 1]

    def extract_endpoints(self, lines: torch.Tensor) -> tuple[list[int], list[int], list[float], list[float]]:
        xmin: list[int] = []
        xmax: list[int] = []
        ymin: list[float] = []
        ymax: list[float] = []
        for line in lines:
            if torch.all(torch.isnan(line)):
                continue
            valid = ~torch.isnan(line)
            start, end = valid.nonzero()[0], valid.nonzero()[-1]
            xmin.append(int(start.item()))
            xmax.append(int(end.item()))
            ymin.append(float(line[start].item()))
            ymax.append(float(line[end].item()))
        return xmin, xmax, ymin, ymax

    def extract_graph_params(self, lines: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        W: int = lines.shape[1]
        xmin, xmax, ymin, ymax = self.extract_endpoints(lines)
        min_coords = torch.tensor(np.column_stack((xmin, ymin)), dtype=torch.float32)
        max_coords = torch.tensor(np.column_stack((xmax, ymax)), dtype=torch.float32)
        heights = torch.nanmean(lines, dim=1).abs()
        return min_coords, max_coords, heights, W

    def compute_cost_matrix(
        self, min_coords: torch.Tensor, max_coords: torch.Tensor, W: int, heights: torch.Tensor
    ) -> tuple[npt.NDArray[Any], torch.Tensor]:
        lam = self.lam
        N = min_coords.shape[0]
        min_exp = min_coords.unsqueeze(1).expand(N, N, 2)
        max_exp = max_coords.unsqueeze(0).expand(N, N, 2)

        delta_x = min_exp[..., 0] - max_exp[..., 0]
        wrapped_x = torch.minimum(delta_x.abs(), W - delta_x.abs()) * lam
        wrapped_mask = (W - delta_x.abs()) < delta_x.abs()

        delta_x = torch.where(delta_x < 0, delta_x / lam, delta_x)
        wrapped_x = torch.where(wrapped_mask, wrapped_x, delta_x.abs())

        delta_y = min_exp[..., 1] - max_exp[..., 1]
        delta_y = torch.where(wrapped_mask, delta_y * lam, delta_y)

        distances = wrapped_x.abs() + delta_y.abs()

        heights_norm = (heights - heights.min()) / heights.max()
        heights_diff = torch.abs(heights_norm.unsqueeze(1) - heights_norm.unsqueeze(0))

        cost_matrix: npt.NDArray[Any] = (distances * (1 + heights_diff * 30)).numpy()
        return cost_matrix, wrapped_mask

    def match_lines(self, cost_matrix: npt.NDArray[Any]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def build_match_graph(self, row_ind: npt.NDArray[Any], col_ind: npt.NDArray[Any]) -> dict[int, list[int]]:
        graph: dict[int, list[int]] = defaultdict(list)
        row_ind = np.atleast_1d(row_ind)
        col_ind = np.atleast_1d(col_ind)
        for i, j in zip(row_ind, col_ind):
            graph[i].append(j)
            graph[j].append(i)
        return graph

    def get_connected_components(self, graph: dict[int, list[int]]) -> list[list[int]]:
        visited: set[int] = set()
        components: list[list[int]] = []
        for node in graph:
            if node in visited:
                continue
            queue = deque([node])
            comp: list[int] = []
            while queue:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)
                comp.append(curr)
                for n in graph[curr]:
                    if n not in visited:
                        queue.append(n)
            components.append(comp)
        return components

    def merge_components(
        self, lines: torch.Tensor, components: list[list[int]]
    ) -> tuple[list[torch.Tensor], list[float]]:
        merged: list[torch.Tensor] = []
        overlaps: list[float] = []
        for group in components:
            group_lines = lines[torch.tensor(group)]
            valid_mask = ~torch.isnan(group_lines)
            merged_line = torch.full((group_lines.shape[1],), float("nan"))
            overlap = valid_mask.sum(0)
            overlaps.append(float(overlap[overlap > 0].float().mean().item()))
            # Later writes win, so visit the component in reverse priority order.
            for row, valid in zip(group_lines.flip(0), valid_mask.flip(0)):
                merged_line = torch.where(valid, row.to(merged_line.dtype), merged_line)
            merged.append(merged_line)
        return merged, overlaps

    def plot_graph(self, min_coords: torch.Tensor, max_coords: torch.Tensor, row_ind: Any, col_ind: Any) -> None:
        plt.figure(figsize=(10, 6))
        for i, j in zip(row_ind, col_ind):
            plt.plot([min_coords[i, 0], max_coords[j, 0]], [min_coords[i, 1], max_coords[j, 1]], color="red")
        plt.scatter(min_coords[:, 0], min_coords[:, 1], color="blue", label="Min Coords")
        plt.scatter(max_coords[:, 0], max_coords[:, 1], color="green", label="Max Coords")
        plt.title("Graph of Merged Lines")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig("sandbox/graph_of_merged_lines.png")
        plt.close()

    def plot_lines(self, lines: torch.Tensor, title: str) -> None:
        plt.figure(figsize=(10, 6))
        for line in lines:
            if torch.all(torch.isnan(line)):
                continue
            plt.plot(line.numpy(), alpha=0.7, linewidth=1.5)
        plt.title(title)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.gca().invert_yaxis()
        plt.savefig(f"sandbox/{title.replace(' ', '_').casefold()}.png")
        plt.close()

    def match_and_merge_lines(self, lines: torch.Tensor) -> tuple[list[torch.Tensor], list[float]]:
        lines = self.preprocess_lines(lines)
        if self.debug:
            self.plot_lines(lines, "Preprocessed Lines")
        min_coords, max_coords, heights, W = self.extract_graph_params(lines)
        cost_matrix, wrapped_mask = self.compute_cost_matrix(min_coords, max_coords, W, heights)

        row_ind, col_ind = self.match_lines(cost_matrix)

        valid_mask = ~wrapped_mask[row_ind, col_ind]
        row_ind, col_ind = row_ind[valid_mask], col_ind[valid_mask]

        graph = self.build_match_graph(row_ind, col_ind)
        components = self.get_connected_components(graph)

        merged_lines, overlaps = self.merge_components(lines, components)
        filtered_lines = [line for line in merged_lines if torch.sum(~torch.isnan(line)) >= W // 5]

        if self.debug:
            self.plot_graph(min_coords, max_coords, row_ind, col_ind)

        return filtered_lines, overlaps
