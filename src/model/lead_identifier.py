import warnings
from copy import deepcopy
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment


class LeadIdentifier:
    LEAD_CHANNEL_ORDER: list[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    def __init__(
        self,
        *,
        layouts: dict[str, Any],
        unet: torch.nn.Module,
        device: torch.device,
        possibly_flipped: bool = True,
        target_num_samples: int = 5000,
        required_valid_samples: int = 3,
        debug: bool = False,
    ) -> None:
        """
        Args:
            layouts: dict mapping layout names to layout definitions.
            unet: a loaded and weight-initialized UNet model (in eval mode).
            device: device where inference will run.
            possibly_flipped: whether to check for flipped layouts (caused by upside-down images)
            target_num_samples: number of samples (seconds*sample rate) that the output should contain.
            required_valid_samples: minimum number of non-NaN samples in a column to consider it valid. Decides where the signal is cropped.
            debug: whether to draw scatter/plots.
        """
        self.layouts = layouts
        self.unet = unet.to(device).eval()
        self.device = device
        self.possibly_flipped = possibly_flipped
        self.target_num_samples = target_num_samples
        self.required_valid_samples = required_valid_samples
        self.debug = debug

    def _merge_nonoverlapping_lines(self, lines: torch.Tensor) -> torch.Tensor:
        if lines.shape[0] > 1:
            means: torch.Tensor = torch.nanmean(lines, dim=1)
            sorted_indices: torch.Tensor = torch.argsort(means)
            lines = lines[sorted_indices]
        changed: bool = True
        while changed and lines.shape[0] > 1:
            changed = False
            new_lines: list[torch.Tensor] = []
            i = 0
            while i < lines.shape[0]:
                if i < lines.shape[0] - 1:
                    row1 = lines[i]
                    row2 = lines[i + 1]
                    overlap = ~(torch.isnan(row1) | torch.isnan(row2))
                    if not torch.any(overlap):
                        merged = torch.where(torch.isnan(row1), row2, row1)
                        new_lines.append(merged)
                        i += 2
                        changed = True
                        continue
                new_lines.append(lines[i])
                i += 1
            lines = torch.stack(new_lines)
        return lines

    def _nan_cossim(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x_values: torch.Tensor = x[~torch.isnan(x)]
        y_values: torch.Tensor = y[~torch.isnan(y)]
        if x_values.numel() <= 1 or y.numel() <= 1:
            return -1.0
        x_mean: torch.Tensor = x_values.mean()
        y_mean: torch.Tensor = y_values.mean()
        x_norm: torch.Tensor = x - x_mean
        y_norm: torch.Tensor = y - y_mean
        both_exist_mask: torch.Tensor = ~(torch.isnan(x_norm) | torch.isnan(y_norm))
        if not torch.any(both_exist_mask):
            return -1.0
        x_norm = x_norm[both_exist_mask]
        y_norm = y_norm[both_exist_mask]
        x_norm = x_norm / torch.linalg.norm(x_norm, keepdim=True)
        y_norm = y_norm / torch.linalg.norm(y_norm, keepdim=True)
        return float(torch.dot(x_norm, y_norm).item())

    def _canonicalize_lines(self, lines: torch.Tensor, match: dict[str, Any]) -> torch.Tensor:
        """Reorders and flips lines to a canonical 12-lead format.

        Args:
            lines: Tensor of lines.
            match: Layout match info.

        Returns:
            Canonicalized tensor.
        """
        canonical_order: list[str] = self.LEAD_CHANNEL_ORDER
        num_leads: int = len(canonical_order)
        width: int = lines.shape[1]
        canonical: torch.Tensor = torch.full((num_leads, width), float("nan"), dtype=lines.dtype, device=lines.device)

        layout_name: Optional[str] = match.get("layout")
        flip: bool = match.get("flip", False)
        if layout_name is None or layout_name not in self.layouts:
            return canonical
        layout_def: dict[str, Any] = self.layouts[layout_name]
        leads_def: Any = layout_def["leads"]
        rhythm_leads: list[str] = layout_def.get("rhythm_leads", [])
        cols: int = layout_def["layout"]["cols"]

        if flip:
            lines = torch.flip(lines, dims=[0, 1])
            valid_lines = lines[~torch.isnan(lines)]
            if valid_lines.numel() > 0:
                lines_max_val = valid_lines.max()
            else:
                lines_max_val = 1.0  # type: ignore
            lines = lines_max_val - lines

        lead_names: list[str] = []
        for row in leads_def:
            if isinstance(row, list):
                lead_names.extend(row)
            else:
                lead_names.append(row)
        lead_names += rhythm_leads

        used_indices: set[tuple[int, int, int]] = set()

        chunk_width: int = width // cols
        for row_idx, layout_row in enumerate(leads_def):
            if not isinstance(layout_row, list):
                layout_row = [layout_row]
            for col_idx, layout_lead in enumerate(layout_row):
                start: int = col_idx * chunk_width
                end: int = (col_idx + 1) * chunk_width if col_idx < cols - 1 else width
                if row_idx >= lines.shape[0]:
                    continue
                chunk: torch.Tensor = lines[row_idx, start:end]
                sign: int = 1
                lead_name: str = layout_lead
                if isinstance(lead_name, str) and lead_name.startswith("-"):
                    sign = -1
                    lead_name = lead_name[1:]
                if lead_name in canonical_order:
                    canon_idx: int = canonical_order.index(lead_name)
                    canonical[canon_idx, start:end] = sign * chunk
                    used_indices.add((canon_idx, start, end))

        # If there are any rythm, leads, we try to match them with the canonical leads, through cosine similarity.
        num_rhythm_leads: int = len(rhythm_leads)
        if num_rhythm_leads > 0:
            rhythm_corrs: NDArray[np.float32] = np.full((num_rhythm_leads, num_leads), -1, dtype=np.float32)
            for i in range(num_rhythm_leads):
                rhythm_vec: torch.Tensor = lines[-num_rhythm_leads + i, :]
                for j in range(num_leads):
                    corr: float = self._nan_cossim(rhythm_vec, canonical[j, :])
                    rhythm_corrs[i, j] = corr

            # Leads II, V1 and V5 are most commonly used for rhythm, so we inflate their cosine similarity.
            # Other matches are still possible.
            if num_rhythm_leads == 1:
                rhythm_corrs[:, 1] = self._inflate_cossim(rhythm_corrs[:, 1])
            elif num_rhythm_leads == 2:
                rhythm_corrs[:, 1] = self._inflate_cossim(rhythm_corrs[:, 1])
                rhythm_corrs[:, 6] = self._inflate_cossim(rhythm_corrs[:, 6])
            elif num_rhythm_leads == 3:
                rhythm_corrs[:, 1] = self._inflate_cossim(rhythm_corrs[:, 1])
                rhythm_corrs[:, 6] = self._inflate_cossim(rhythm_corrs[:, 6])
                rhythm_corrs[:, 10] = self._inflate_cossim(rhythm_corrs[:, 10])
            try:
                row_idx, col_idx = linear_sum_assignment(-rhythm_corrs)
                for i_r, i_c in zip(row_idx, col_idx):
                    corr_val: float = rhythm_corrs[i_r, i_c]
                    print(f"Rhythm {i_r} → Canonical {canonical_order[i_c]} (corr={corr_val:.2f})")
                    canonical[i_c, :] = lines[-num_rhythm_leads + i_r, :]
            except ValueError:
                if self.debug:
                    print("Linear sum assignment failed, possibly due to NaN values in rhythm correlations.")

        return canonical

    def _inflate_cossim(
        self, lead: Union[float, NDArray[Any], torch.Tensor], factor: float = 0.75
    ) -> Union[float, NDArray[Any], torch.Tensor]:
        return 1 - factor + factor * lead

    def _replace_duplicate_leads_layouts(self, layout_def: dict[str, Any]) -> dict[str, Any]:
        """
        For each row in layout, replace all but the first duplicate with X
        """
        output_layout_def = deepcopy(layout_def)
        for row in output_layout_def["leads"]:
            seen = set()
            if not isinstance(row, list):
                continue
            for i, lead in enumerate(row):
                if lead in seen:
                    row[i] = "X"
                else:
                    seen.add(lead)
        return output_layout_def

    def _generate_grid_positions(self, layout_def: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
        """Generates normalized grid positions for each lead.

        Args:
            layout_def: Layout definition dict.

        Returns:
            Mapping of lead name to grid position array.
        """
        clean_layout_def = self._replace_duplicate_leads_layouts(layout_def)
        rows: int = clean_layout_def["layout"]["rows"]
        cols: int = clean_layout_def["layout"]["cols"]
        leads: Any = clean_layout_def["leads"]

        def norm_y(i: int) -> float:
            return i / (rows - 1) if rows > 1 else 0.5

        def norm_x(j: int) -> float:
            return j / (cols - 1) if cols > 1 else 0.5

        pos: dict[str, NDArray[np.float64]] = {}
        lead_str: str
        if isinstance(leads[0], list):
            for y_idx, row in enumerate(leads):
                for x_idx, lead in enumerate(row):
                    lead_str = lead.strip("-")
                    if len(pos.get(lead_str, np.array([]))) == 0:
                        pos[lead_str] = np.array([norm_x(x_idx), norm_y(y_idx)], dtype=np.float64)
                    else:
                        pos[lead_str] = (pos[lead_str] + np.array([norm_x(x_idx), norm_y(y_idx)], dtype=np.float64)) / 2
        elif len(leads) == rows * cols:
            for idx, lead in enumerate(leads):
                y_idx, x_idx = divmod(idx, cols)
                lead_str = lead.strip("-")
                if len(pos.get(lead_str, np.array([]))) == 0:
                    pos[lead_str] = np.array([norm_x(x_idx), norm_y(y_idx)], dtype=np.float64)
                else:
                    pos[lead_str] = (pos[lead_str] + np.array([norm_x(x_idx), norm_y(y_idx)], dtype=np.float64)) / 2
        else:
            for y_idx, lead in enumerate(leads):
                lead_str = lead.strip("-")
                if len(pos.get(lead_str, np.array([]))) == 0:
                    pos[lead_str] = np.array([0.5, norm_y(y_idx)], dtype=np.float64)
                else:
                    pos[lead_str] = (pos[lead_str] + np.array([0.5, norm_y(y_idx)], dtype=np.float64)) / 2
        # drop the X key if it exists (placeholder for duplicate leads)
        if "X" in pos:
            del pos["X"]
        return pos

    def _extract_lead_points(
        self, probs_tensor: torch.Tensor, lead_names: Optional[list[str]] = None
    ) -> list[tuple[str, float, float]]:
        if lead_names is None:
            lead_names = self.LEAD_CHANNEL_ORDER
        _, C, H, W = probs_tensor.shape
        arr: NDArray[Any] = probs_tensor[0].cpu().numpy()
        pts: list[tuple[str, float, float]] = []
        for i, name in enumerate(lead_names):
            channel: NDArray[Any] = arr[i]
            if np.sum(channel) == 0:
                continue
            x_fmap: NDArray[np.float64] = np.arange(W).reshape(1, W).repeat(H, axis=0).astype(np.float64)
            y_fmap: NDArray[np.float64] = np.arange(H).reshape(H, 1).repeat(W, axis=1).astype(np.float64)
            x_com: float = float(np.sum(x_fmap * channel) / np.sum(channel))
            y_com: float = float(np.sum(y_fmap * channel) / np.sum(channel))
            pts.append((name, x_com, y_com))
        return pts

    def _check_cabrera_limb(self, ts: NDArray[np.float64]) -> tuple[float, float]:
        B = np.array(
            [
                [1, -0.5],  # aVL = I - 0.5 II
                [1, 0],  # I
                [0.5, 0.5],  # -aVR = 0.5 I + 0.5 II
                [0, 1],  # II
                [-0.5, 1],  # aVF = -0.5 I + II
                [-1, 1],  # III = -I + II
            ]
        )
        A = np.linalg.pinv(B)

        compressed_ts = ts.T @ B  # shape (channels, 6)
        reconstructed_ts = (compressed_ts @ A).T  # shape (6, time)
        cossim = self._nan_cossim(
            torch.from_numpy(reconstructed_ts.flatten()).float(), torch.from_numpy(ts.flatten()).float()
        )

        flipped_ts = np.flip(np.flip(ts, axis=0), axis=1)
        flipped_compressed_ts = flipped_ts.T @ B
        flipped_reconstructed_ts = (flipped_compressed_ts @ A).T
        flipped_cossim = self._nan_cossim(
            torch.from_numpy(flipped_reconstructed_ts.flatten()).float(), torch.from_numpy(flipped_ts.flatten()).float()
        )

        return cossim, flipped_cossim

    def _match_layout(
        self,
        detected_pts: list[tuple[str, float, float]],
        rows_in_layout: int,
        layouts: dict[str, Any],
        check_flipped: bool,
    ) -> dict[str, Any]:
        names, xs, ys = zip(*detected_pts)
        pts: NDArray[np.float64] = np.stack([xs, ys], axis=1)
        n: int = pts.shape[0]
        best: dict[str, Any] = {"cost": np.inf}

        for layout_name, desc in layouts.items():
            total_rows: int = desc["total_rows"]
            rows_difference: int = abs(total_rows - rows_in_layout)
            pos_map: dict[str, NDArray[np.float64]] = self._generate_grid_positions(desc)
            if "I" in pos_map:
                del pos_map["I"]
            grid_leads: list[str] = list(pos_map.keys())
            grid_pts: NDArray[np.float64] = np.stack([pos_map[lead] for lead in grid_leads])

            flip_options = (False, True) if check_flipped else (False,)

            for flip in flip_options:
                scaling_factor: float = max(len(grid_leads), n) / min(len(grid_leads), n) * (1 + rows_difference * 3)
                P: NDArray[np.float64] = pts.copy()
                if flip:
                    P = -P

                Pm: list[NDArray[np.float64]] = []
                Gm: list[NDArray[np.float64]] = []
                idxs: list[tuple[int, int]] = []
                missing: int = 0
                for i, lead in enumerate(names):
                    if lead in pos_map:
                        j = grid_leads.index(lead)
                        Pm.append(P[i])
                        Gm.append(grid_pts[j])
                        idxs.append((i, j))
                    else:
                        missing += 1
                Pm_arr: NDArray[np.float64] = np.array(Pm)
                Gm_arr: NDArray[np.float64] = np.array(Gm)
                if Pm_arr.shape[0] < 2:
                    continue

                mu_P: NDArray[np.float64] = Pm_arr.mean(axis=0)
                mu_G: NDArray[np.float64] = Gm_arr.mean(axis=0)
                Pc: NDArray[np.float64] = Pm_arr - mu_P
                Gc: NDArray[np.float64] = Gm_arr - mu_G
                num: NDArray[np.float64] = np.sum(Pc * Gc, axis=0)
                den: NDArray[np.float64] = np.sum(Pc**2, axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    s: NDArray[np.float64] = num / den
                s = np.where(np.isfinite(s), s, 0.0)
                if np.any(s < 0):
                    scaling_factor *= 2
                s[s < 1e-4] = 1e-4
                t: NDArray[np.float64] = mu_G - s * mu_P
                P_scaled: NDArray[np.float64] = P * s + t

                res: list[float] = []
                for i, j in idxs:
                    res.append(float(np.linalg.norm(P_scaled[i] - grid_pts[j])))
                PENALTY: float = 0.5
                res.extend([PENALTY] * missing)
                avg_res: float = float(np.mean(res)) * scaling_factor

                if avg_res < best["cost"]:
                    best = {"layout": layout_name, "flip": flip, "cost": avg_res, "leads": grid_leads}
                if self.debug:
                    for ii, jj in idxs:
                        plt.plot(
                            [P_scaled[ii, 0], grid_pts[jj, 0]],
                            [P_scaled[ii, 1], grid_pts[jj, 1]],
                            c="gray",
                            alpha=0.5,
                        )
                    plt.scatter(grid_pts[:, 0], grid_pts[:, 1], c="blue", label="Layout", s=250)
                    for j, lead in enumerate(grid_leads):
                        plt.text(
                            grid_pts[j, 0], grid_pts[j, 1], lead, ha="center", va="center", fontsize=8, color="white"
                        )
                    plt.scatter(P_scaled[:, 0], P_scaled[:, 1], c="red", label="Detected", s=250)
                    for i, name in enumerate(names):
                        plt.text(
                            P_scaled[i, 0], P_scaled[i, 1], name, ha="center", va="center", fontsize=8, color="white"
                        )
                    plt.gca().invert_yaxis()
                    plt.title(f"{layout_name} (flip={flip}) - cost={avg_res:.4f}")
                    plt.legend()
                    plt.savefig(f"sandbox/match_debug_{layout_name}_{flip}.png", dpi=200)
                    plt.close()
        return best

    def _interpolate_lines(self, lines: torch.Tensor, target_num_samples: int) -> torch.Tensor:
        if lines.shape[1] == target_num_samples:
            return lines

        num_leads: int = lines.shape[0]

        x: NDArray[np.floating] = np.linspace(0, 1, num=lines.shape[1])
        x_new: NDArray[np.floating] = np.linspace(0, 1, num=target_num_samples)
        interpolated_lines: list[torch.Tensor] = []
        for i in range(num_leads):
            lead_line: NDArray[np.float64] = lines[i].cpu().numpy()
            interpolated_line: NDArray[np.float64] = np.interp(x_new, x, lead_line)
            interpolated_lines.append(torch.tensor(interpolated_line, dtype=lines.dtype, device=lines.device))
        if len(interpolated_lines) == 0:
            return torch.empty((num_leads, target_num_samples), dtype=lines.dtype, device=lines.device)
        return torch.stack(interpolated_lines)

    def normalize(self, lines: torch.Tensor, avg_pixel_per_mm: float, mv_per_mm: float) -> torch.Tensor:
        """Changes the units of the ECG signals from pixels to mV/mm."""
        lines = lines - lines.nanmean(dim=1, keepdim=True)
        lines = lines * (mv_per_mm / avg_pixel_per_mm) * 1000

        non_nan_samples_per_column = torch.sum(~torch.isnan(lines), dim=0).numpy()
        first_valid_index: int = int(np.argmax(non_nan_samples_per_column >= self.required_valid_samples))
        last_valid_index: int = int(np.argmax(non_nan_samples_per_column[::-1] >= self.required_valid_samples))
        last_valid_index = lines.shape[1] - last_valid_index - 1
        if first_valid_index <= last_valid_index:
            lines = lines[:, first_valid_index : last_valid_index + 1]

        lines = self._interpolate_lines(lines, self.target_num_samples)

        return lines

    def _restrict_cabrera_layouts_if_needed(
        self, lines: torch.Tensor, layouts: dict[str, Any], possibly_flipped: bool
    ) -> tuple[dict[str, Any], bool]:
        should_check_flipped = possibly_flipped
        if lines.shape[0] == 6:
            cossim, cossim_flipped = self._check_cabrera_limb(lines.cpu().numpy())
            max_cossim: float = max(cossim, cossim_flipped)
            if max_cossim > 0.992:
                layouts = {
                    name: desc for name, desc in layouts.items() if "cabrera" in name.lower() and "limb" in name.lower()
                }
            elif max_cossim < 0.95:
                layouts = {
                    name: desc
                    for name, desc in layouts.items()
                    if "cabrera" not in name.lower() or "limb" not in name.lower()
                }
            if cossim_flipped == max_cossim:
                should_check_flipped = True
        return layouts, should_check_flipped

    def __call__(
        self,
        lines: torch.Tensor,
        feature_map: torch.Tensor,
        avg_pixel_per_mm: float,
        threshold: float = 0.8,
        mv_per_mm: float = 0.1,
        layout_should_include_substring: Optional[str] = None,
    ) -> dict[str, Any]:
        lines = self._merge_nonoverlapping_lines(lines)
        lines = -self.normalize(lines, avg_pixel_per_mm, mv_per_mm)
        layouts = self.layouts.copy()

        if layout_should_include_substring is not None:
            layouts = {
                name: desc for name, desc in layouts.items() if layout_should_include_substring.lower() in name.lower()
            }

        # layouts, check_flipped = self._restrict_cabrera_layouts_if_needed(lines, layouts, self.possibly_flipped)

        rows_in_layout: int = lines.shape[0]
        self.unet.eval()
        with torch.no_grad():
            logits: torch.Tensor = self.unet(feature_map.to(self.device))  # [1,13,H,W]
            probs: torch.Tensor = torch.softmax(logits, dim=1)[:, :12]  # [1,12,H,W]
            probs[:, 0] = 0  # Ignore the position of the "I" lead as it is particularly prone to false positives.
        probs[probs < threshold] = 0
        detected: list[tuple[str, float, float]] = self._extract_lead_points(probs, self.LEAD_CHANNEL_ORDER)
        if len(detected) <= 2:
            match: dict[str, Any] = {"cost": float("inf")}
            canonical_lines: Optional[torch.Tensor] = None
        else:
            match = self._match_layout(detected, rows_in_layout, layouts, self.possibly_flipped)

        if "layout" not in match:
            if not layouts:
                # Caller over-constrained the candidate set (e.g. a layout hint
                # that doesn't match any entry in the YAML). Fail loudly rather
                # than silently mislabeling leads.
                raise ValueError(
                    "No candidate layouts available for lead matching. Check "
                    "that `layout_should_include_substring` matches an entry "
                    "in the lead-layouts YAML."
                )
            warnings.warn(
                f"No matching layout found (n_detected={len(detected)}, "
                f"rows_in_layout={rows_in_layout}). Falling back to first "
                f"candidate '{list(layouts.keys())[0]}' — canonical lead "
                f"assignments are unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
            match["layout"] = list(layouts.keys())[0]

        canonical_lines = self._canonicalize_lines(lines.clone(), match)

        return {
            "rows_in_layout": rows_in_layout,
            "n_detected": len(detected),
            **match,
            "canonical_lines": canonical_lines,
            "lines": lines,
        }
