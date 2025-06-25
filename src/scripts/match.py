import os
import sys

# Add the parent of the project root to sys.path
sys.path.append(os.path.abspath("../../"))

import pandas as pd
import torch
import torchvision
from PIL import Image
from torchaudio.transforms import Resample

from src.config.default import get_cfg
from src.utils import load_model

config = get_cfg("src/config/inference_wrapper.yml")
inference_wrapper = load_model(config)


ts_folder = "/home/datasets/ecg-digitization/csvs/"
ts_dfs_dict = {}

for file_name in os.listdir(ts_folder):
    file_path = os.path.join(ts_folder, file_name)
    ts_dfs_dict[file_name] = pd.read_csv(file_path)

ts_df = pd.concat(ts_dfs_dict.values(), keys=ts_dfs_dict.keys(), names=["file_name", "index"]).reset_index()
ts_df["file_name"] = ts_df["file_name"].astype("category")


def autocorr_with_nans_vectorized(x, y, lags):
    """
    Vectorized cross-correlation with NaN handling across multiple lags.

    Args:
        x (torch.Tensor): shape (T,)
        y (torch.Tensor): shape (T,)
        lags (list or torch.Tensor): shape (L,)

    Returns:
        torch.Tensor: shape (L,), correlation coefficients for each lag
    """
    x = x.clone()
    y = y.clone()
    lags = torch.tensor(lags, dtype=torch.long)

    T = x.shape[0]
    L = lags.shape[0]

    # Allocate index tensors for each lag
    indices = torch.arange(T).expand(L, T)  # shape: (L, T)
    lagged_indices_x = indices.clone()

    lagged_indices_x = lagged_indices_x - lags.view(-1, 1)  # each row i: indices - lags[i]

    # Mask out-of-bounds indices
    valid_mask = (lagged_indices_x >= 0) & (lagged_indices_x < T)

    # Gather values (invalid indices will be masked out later)
    x_lagged = torch.where(valid_mask, x[lagged_indices_x.clamp(0, T - 1)], torch.tensor(float("nan"), device=x.device))
    y_batched = y.unsqueeze(0).expand(L, y.shape[0])

    num_points = y_batched.shape[-1]
    if x_lagged.shape[-1] > num_points:
        start = x_lagged.shape[-1] // 2 - num_points // 2
        end = start + num_points
        x_lagged = x_lagged[:, start:end]
        valid_mask = valid_mask[:, start:end]
    elif x_lagged.shape[-1] < num_points:
        padding = num_points - x_lagged.shape[-1]
        x_lagged = torch.nn.functional.pad(x_lagged, (0, padding))
        valid_mask = torch.nn.functional.pad(valid_mask, (0, padding), value=False)

    # Mask NaNs
    combined_mask = valid_mask & ~torch.isnan(x_lagged) & ~torch.isnan(y_batched)

    x_valid = torch.where(combined_mask, x_lagged, torch.tensor(0.0, device=x.device))
    y_valid = torch.where(combined_mask, y_batched, torch.tensor(0.0, device=y.device))
    count_valid = combined_mask.sum(dim=1)

    # Mean centering
    x_sum = x_valid.sum(dim=1)
    y_sum = y_valid.sum(dim=1)

    x_mean = x_sum / count_valid.clamp(min=1)
    y_mean = y_sum / count_valid.clamp(min=1)

    x_centered = x_valid - x_mean.unsqueeze(1)
    y_centered = y_valid - y_mean.unsqueeze(1)

    numerator = (x_centered * y_centered * combined_mask).sum(dim=1)
    denominator = torch.sqrt((x_centered**2 * combined_mask).sum(dim=1) * (y_centered**2 * combined_mask).sum(dim=1))

    corr = numerator / denominator
    corr[count_valid < 2] = float("nan")  # not enough data

    return corr


def find_max_correlation(image_signal, ts, ts_sample_rate):
    max_offset_time = 0.5
    max_offset_points = int(max_offset_time * ts_sample_rate)
    offsets = list(range(-max_offset_points, max_offset_points, 50))

    correlation = autocorr_with_nans_vectorized(image_signal, ts, offsets)
    return correlation.max(), (correlation.argmax() * 50 - max_offset_points) / ts_sample_rate


# import plotly.graph_objects as go
# from IPython.display import display
#
# scatter = go.Scatter(
#    x=[], y=[], mode='markers',
#    text=[], textposition='top center', hoverinfo='text+x+y',
#    marker=dict(size=10)
# )
#
# fig = go.FigureWidget(data=[scatter])
# display(fig)
correlations = []
correlation_indices = []
names = []

# Either:
#  - V1, V2, V3, V4, V5, V6
#  - aVL, I, -aVR, II, aVF, III


save_matches_dir = "/home/datasets/ecg-digitization/matched/agnar"
# remove old matches


if os.path.exists(save_matches_dir):
    import shutil

    shutil.rmtree(save_matches_dir)

os.makedirs(save_matches_dir, exist_ok=True)

image_directory = "/home/datasets/ecg-digitization/redacted/agnar_digital_redactions/"

num_matches = 0
MIN_CORRELATION_TRESHOLD = 0.92

import numpy as np
import tqdm

for image_file_name in tqdm.tqdm(os.listdir(image_directory)):
    if not image_file_name.endswith(".JPG"):
        continue
    if "2100" not in image_file_name:
        continue
    image_path = os.path.join(image_directory, image_file_name)
    print(f"Processing {image_path}")

    img = torchvision.io.read_image(image_path)
    res = inference_wrapper(img.unsqueeze(0))

    image_pixels_per_mm = inference_wrapper.dewarper.pixels_per_mm
    image_mm_per_second = 50
    signal_length_seconds = 10

    image_pixels_per_second = image_pixels_per_mm * image_mm_per_second
    image_signal_length_pixels = image_pixels_per_mm * image_mm_per_second * signal_length_seconds

    best_correlation = -1
    best_correlation_name = None
    best_correlation_idx = None
    best_channel = None

    image_ts = res["snake"]
    ts_sample_rate = 10000

    for s in [["V1", "V2", "V3", "V4", "V5", "V6"], ["aVL", "I", "aVR", "II", "aVF", "III"]]:
        for file_name in ts_df["file_name"].unique():
            if "161_" not in file_name:
                continue
            ts = ts_df.loc[ts_df["file_name"] == file_name]
            curr_correlations = []
            for idx, v in enumerate(s):
                ts_channel = ts[v].values

                if v == "aVR":
                    ts_channel = -ts_channel

                ts_signal_length_points = ts_channel.shape[0]
                ts_sample_rate = ts_signal_length_points // signal_length_seconds

                # resampled_image_signal = Resample(orig_freq=int(image_pixels_per_second), new_freq=ts_sample_rate)(
                #    image_ts[idx % 6]
                # )
                resampled_image_signal = np.interp(
                    np.arange(0, ts_sample_rate), np.arange(0, image_ts[idx % 6].shape[0]), image_ts[idx % 6]
                )
                correlation, correlation_idx = find_max_correlation(
                    torch.tensor(resampled_image_signal), torch.tensor(ts_channel[:5000]), ts_sample_rate
                )
                # do not correlation that is nan.
                if torch.isnan(correlation):
                    continue
                curr_correlations.append(correlation)

            print(
                f"File: {file_name}, Correlation: {np.mean(np.array(curr_correlations))}, seconds off: {correlation_idx}"
            )
            correlation = np.mean(np.array(curr_correlations))
            if correlation > best_correlation:
                best_correlation = correlation
                best_correlation_name = file_name
                best_correlation_idx = correlation_idx
                best_channel = s[0]
            # print(
            #    f"new best correlation file: {file_name}, Correlation: {correlation}, seconds off: {correlation_idx}"
            # )
        print(
            f"File: {best_correlation_name}, Max Correlation: {best_correlation}, seconds off: {best_correlation_idx}"
        )
        if best_correlation > MIN_CORRELATION_TRESHOLD:
            num_matches += 1
            save_path = os.path.join(save_matches_dir, image_file_name[:-4])
            os.makedirs(save_path, exist_ok=True)

            if best_channel == "V1":
                channels = ["V1", "V2", "V3", "V4", "V5", "V6"]
            else:
                channels = ["aVL", "I", "-aVR", "II", "aVF", "III"]

            ts_df[ts_df["file_name"] == best_correlation_name][channels].to_csv(
                os.path.join(save_path, best_correlation_name), index=False
            )

            print(
                f"Best file: {best_correlation_name}, Correlation: {best_correlation}, seconds off: {best_correlation_idx}"
            )
            print("total num matches:", num_matches)
