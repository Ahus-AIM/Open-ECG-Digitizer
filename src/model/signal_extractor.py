from collections import defaultdict, deque
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torchvision
from scipy.optimize import linear_sum_assignment

from src.seed import image_generator


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
            return torch.empty((0, fmap.shape[1]), dtype=torch.float32)
        lines = torch.stack(lines_list, dim=0)
        merged_lines_list, overlaps = self.match_and_merge_lines(lines)
        if len(merged_lines_list) == 0:
            return torch.empty((0, fmap.shape[1]), dtype=torch.float32)
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
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[slice, slice, torch.Tensor]]]:
        labeled = self._stripe_labels(fmap)
        if labeled is None:
            return [], [], []
        lab, ys, xs = labeled
        pixels = fmap.numpy()[ys, xs]
        height, width = lab.shape
        good: list[torch.Tensor] = []
        rejected: list[torch.Tensor] = []
        rej_maps: list[tuple[slice, slice, torch.Tensor]] = []
        # A column reduction is independent of all other columns. Pack disjoint
        # (label, column) pairs together while preserving the full reduction
        # shape, dtype and row order of the original masked sums.
        pixel_labels = lab[ys, xs]
        stride = int(pixel_labels.max()) + 1
        pairs, inverse = np.unique(xs.astype(np.int64) * stride + pixel_labels, return_inverse=True)
        cols, labels = np.divmod(pairs, stride)
        starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
        slots = np.arange(len(cols)) - np.repeat(starts, np.diff(starts, append=len(cols)))
        pixel_slots = slots[inverse]
        centers = torch.empty(len(cols), dtype=fmap.dtype)
        masked = torch.zeros((height, width), dtype=fmap.dtype)
        buffer = masked.numpy()
        for slot in range(int(slots.max()) + 1):
            active = pixel_slots == slot
            y, x, v = ys[active], xs[active], pixels[active]
            buffer[y, x] = v
            mass = masked.sum(0).clamp_min_(1e-6).numpy()
            buffer[y, x] = (v / mass[x]) * y.astype(v.dtype)
            line = masked.sum(0)
            line[line < 1 / height] = float("nan")
            selected = slots == slot
            centers[selected] = line[cols[selected]]
            buffer[y, x] = 0

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
                region_rows = ys[pixel_labels == lid]
                y = slice(int(region_rows.min()), int(region_rows.max()) + 1)
                x = slice(int(active_cols[0]), int(active_cols[-1]) + 1)
                rej_maps.append((y, x, fmap[y, x] * torch.from_numpy(lab[y, x] == lid)))

        for line in (*good, *rejected):
            line[line < 5] = float("nan")

        return good, rejected, rej_maps

    def _stripe_labels(
        self, fmap: torch.Tensor
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]] | None:
        # SAUF numbers four-connected components in row-major order. This
        # order also determines priority when fragments are later merged.
        foreground = (fmap.numpy() > self.label_thresh).view(np.uint8)
        points = cv2.findNonZero(foreground)
        if points is None:
            return None
        xs, ys = np.asarray(points, dtype=np.int32).reshape(-1, 2).T
        count, regions = cv2.connectedComponentsWithAlgorithm(foreground, 4, cv2.CV_32S, cv2.CCL_SAUF)
        lab = np.asarray(regions, dtype=np.int32)
        labels = lab[ys, xs]
        # Stable grouping preserves the pixel order of each original masked
        # Torch sum, including for components right at the rejection threshold.
        order = np.argsort(labels, kind="stable")
        pixels = torch.from_numpy(fmap.numpy()[ys[order], xs[order]])
        ends = np.bincount(labels, minlength=count).cumsum()
        keep = np.zeros(count, dtype=bool)
        for lid in range(1, count):
            keep[lid] = not bool(pixels[ends[lid - 1] : ends[lid]].sum() < self.threshold_sum)
        valid = keep[labels]
        if not valid.all():
            foreground[ys[~valid], xs[~valid]] = 0
            lab[ys[~valid], xs[~valid]] = 0
            xs, ys, labels = xs[valid], ys[valid], labels[valid]
        if not xs.size:
            return None
        global_ids = keep.cumsum(dtype=np.int32)[labels]
        width = foreground.shape[1]
        big_offset = int(global_ids.max()) + 1000
        for stripe in range(self.split_num_stripes):
            start, stop = stripe * width // self.split_num_stripes, (stripe + 1) * width // self.split_num_stripes
            if stop > start:
                _, local = cv2.connectedComponentsWithAlgorithm(foreground[:, start:stop], 4, cv2.CV_32S, cv2.CCL_SAUF)
                selected = (xs >= start) & (xs < stop)
                y, x = ys[selected], xs[selected]
                lab[y, x] = global_ids[selected] + big_offset * stripe + local[y, x - start]
        if self.split_num_stripes <= 0:
            lab[ys, xs] = global_ids
        return lab, ys, xs

    def _refine_fmap_by_removal(self, fmap: torch.Tensor, rej_maps: list[tuple[slice, slice, torch.Tensor]]) -> None:
        for y, x, mask_map in rej_maps:
            ys, xs = torch.where(mask_map > 0)
            if ys.numel() == 0:
                continue
            y0, y1 = int(ys.min().item()), int(ys.max().item())
            x0, x1 = int(xs.min().item()), int(xs.max().item())
            cropped: npt.NDArray[Any] = mask_map[y0 : y1 + 1, x0 : x1 + 1].cpu().numpy()
            _, new_img = self._trace_horizontal_path(cropped)
            region = fmap[y, x][y0 : y1 + 1, x0 : x1 + 1]
            region[cropped > 0] = new_img[cropped > 0]

    def _trace_horizontal_path(self, img_arr: npt.NDArray[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.tensor(img_arr, dtype=torch.float32)
        height, width = img.shape
        blurry = img + (torch.linspace(-1, 1, height) ** 6)[:, None]
        if width == 0:
            return torch.tensor([height // 2]), img
        span = self.candidate_span
        offsets = torch.arange(-span, span + 1)
        penalty = torch.tensor(np.linspace(-1, 1, len(offsets)) ** 2, dtype=torch.float32)
        noise = torch.empty((width - 1, len(offsets)))
        for row in noise:
            torch.randn(len(offsets), generator=image_generator(), out=row)
        noise *= 1e-6
        path = np.full(width, height // 2, dtype=np.int64)

        # Estimate the walk from vertical prefix sums, then check the proposed
        # transitions together using the original floating-point reductions.
        padded = np.pad(blurry.numpy(), ((span, span), (0, 0)), mode="edge")
        prefix = np.zeros((len(padded) + 1, width), dtype=np.float64)
        np.cumsum(padded, axis=0, out=prefix[1:])
        pairs = ((prefix[:, :-1] + prefix[:, 1:]) * 0.5).T.copy()
        lengths = np.abs(offsets.numpy()) + 1
        lows = span + np.minimum(offsets.numpy(), 0)
        highs = span + np.maximum(offsets.numpy(), 0) + 1
        jitter, weights = noise.numpy(), penalty.numpy()
        first = 0
        while first < width - 1:
            stop = min(width - 1, first + 256)
            for x in range(first, stop):
                prev = path[x]
                scores = ((pairs[x, prev + highs] - pairs[x, prev + lows]) / lengths).astype(np.float32) + jitter[x]
                centered = scores - scores.mean()
                dispersion = np.sqrt((centered @ centered) / max(1, len(scores) - 1))
                move = int((scores + dispersion * weights).argmin()) - span
                path[x + 1] = min(height - 1, max(0, prev + move))

            prev_rows = torch.from_numpy(path[first:stop])
            rows = (prev_rows[:, None] + offsets).clamp(0, height - 1)
            cols = torch.arange(first, stop)[:, None]
            left, right = blurry[rows, cols], blurry[rows, cols + 1]
            exact = torch.empty((stop - first, len(offsets)))
            for i in range(len(offsets)):
                sl = slice(min(i, span), max(i, span) + 1)
                exact[:, i] = (left[:, sl].contiguous().mean(1) + right[:, sl].contiguous().mean(1)) / 2
            exact += noise[first:stop]
            exact += exact.std(1, keepdim=True) * penalty
            correct = (prev_rows + exact.argmin(1) - span).clamp(0, height - 1).numpy()
            mismatch = np.flatnonzero(correct != path[first + 1 : stop + 1])
            if mismatch.size:
                first += int(mismatch[0]) + 1
                path[first] = correct[mismatch[0]]
            else:
                first = stop

        for i, (left_y, right_y) in enumerate(zip(path, path[1:])):
            lo, hi = sorted((left_y, right_y))
            img[lo - 1 : hi + 2, i] = 0
        return torch.from_numpy(path), img

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
        if tensor.dtype == torch.float32 and tensor.device.type == "cpu" and 1 < kernel_size < 2 * min(tensor.shape):
            # Use the same discrete Gaussian kernel and reflected border as torchvision.
            half = (kernel_size - 1) / 2
            axis = torch.linspace(-half, half, kernel_size, dtype=tensor.dtype)
            weights = torch.exp(-0.5 * (axis / sigma).square())
            weights /= weights.sum()
            kernel = torch.outer(weights, weights).numpy()
            return torch.from_numpy(cv2.filter2D(tensor.numpy(), -1, kernel, borderType=cv2.BORDER_REFLECT_101))
        blurred_tensor: torch.Tensor = (
            torchvision.transforms.functional.gaussian_blur(
                tensor[None, None], [kernel_size, kernel_size], [sigma, sigma]
            )
            .squeeze(0)
            .squeeze(0)
        )
        return blurred_tensor

    def _autodetect_num_peaks(self, fmap: torch.Tensor) -> int:
        clamped = fmap.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(1 - clamped) / torch.log(clamped)
        logits = self._blur(logits)
        probs_raw = torch.softmax(logits, dim=0)
        zero_mask = probs_raw <= 1.1 / probs_raw.shape[0]
        lines = zero_mask[1:] & ~zero_mask[:-1]
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
