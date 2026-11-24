from __future__ import annotations

import argparse
import multiprocessing
import os
import random
import warnings
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

from tqdm import tqdm
from yacs.config import CfgNode as CN

from src.config.default import find_config_path, get_cfg, import_class_from_path
from src.seed import DEFAULT_INFERENCE_SEED, seed_inference

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch

    from src.plot_ecg import ECGPlot

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_candidate_file_paths(config: CN) -> list[str]:
    candidate_filenames: list[str] = []
    for root, _, files in os.walk(config.DATA.images_path):
        for file in files:
            if file.endswith(tuple(config.DATA.image_extensions)):
                candidate_filenames.append(os.path.join(root, file))
    random.seed(DEFAULT_INFERENCE_SEED)
    random.shuffle(candidate_filenames)
    return candidate_filenames


def clear_and_prepare_output_dir(config: CN) -> None:
    output_path: str = config.DATA.output_path
    if os.path.exists(output_path):
        for root, dirs, files in os.walk(output_path, topdown=False):
            if config.DATA.get("clear_output_dir_if_exists", False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_file_structure(config.DATA.images_path, output_path)


def copy_file_structure(src: str, dst: str) -> None:
    for root, dirs, _ in os.walk(src):
        rel_path: str = os.path.relpath(root, src)
        target_dir: str = os.path.join(dst, rel_path)
        os.makedirs(target_dir, exist_ok=True)


def decode_and_prepare_image(file_path: str) -> torch.Tensor:
    from torchvision.io import decode_image

    image: torch.Tensor = decode_image(file_path, mode="RGB")
    C, H, W = image.shape
    if C == 1:
        image = image.expand(3, H, W)
    elif C == 4:
        image = image[:3]
    return image.unsqueeze(0)


def canonical_from_got_values(got_values: dict[str, Any]) -> torch.Tensor | None:
    canonical: torch.Tensor | None = None
    if "signal" in got_values and isinstance(got_values["signal"], dict):
        canonical = got_values["signal"].get("canonical_lines")
    elif "canonical_lines" in got_values:
        canonical = got_values["canonical_lines"]
    return canonical


def save_timeseries_csv(canonical: torch.Tensor | None, output_basepath: str) -> None:
    import numpy as np

    if canonical is None:
        return
    data: npt.NDArray[Any] = canonical.squeeze().cpu().numpy()
    # Assume shape (n_leads, n_points)
    if data.ndim == 1:
        data = data[None, :]
    n_leads = data.shape[0]
    col_names = LEAD_NAMES[:n_leads]
    # Transpose to shape (n_points, n_leads)
    data = data.T
    header = ",".join(col_names)
    np.savetxt(output_basepath + "_timeseries_canonical.csv", data, delimiter=",", header=header, comments="")


def _prepare_plot(got_values: dict[str, Any], canonical: torch.Tensor | None) -> ECGPlot:
    import numpy as np

    from src.plot_ecg import ECGPlot

    spacing = got_values["pixel_spacing_mm"]
    # Convert only after the original scalar arithmetic to retain its dtype.
    return ECGPlot(
        image=got_values["input_image"].squeeze().permute(1, 2, 0).cpu().numpy(),
        aligned=got_values["aligned"]["image"].squeeze().permute(1, 2, 0).cpu().numpy(),
        signal_prob=got_values["aligned"]["signal_prob"].squeeze().cpu().numpy(),
        source_points=got_values["source_points"].cpu().numpy(),
        lines=canonical.squeeze().cpu().numpy() if canonical is not None else None,
        xs=[np.asarray(i * 5 / spacing["x"]).flat[0] for i in range(0, 15, 2)],
        ys=[np.asarray(i * 5 / spacing["y"]).flat[0] for i in range(0, 15, 2)],
        width=np.asarray(5 / spacing["x"]).flat[0],
        height=np.asarray(5 / spacing["y"]).flat[0],
        title=got_values.get("layout_name", "") + f' Layout cost: {got_values["signal"]["layout_matching_cost"]:.2f}',
    )


def save_png_plot(got_values: dict[str, Any], canonical: torch.Tensor | None, output_basepath: str) -> None:
    _prepare_plot(got_values, canonical).save(output_basepath)


def save_matching_cost(got_values: dict[str, Any], output_basepath: str) -> None:
    _append_metadata(output_basepath, _metadata_row(got_values, output_basepath))


def _metadata_row(got_values: dict[str, Any], output_basepath: str) -> str:
    name = os.path.basename(output_basepath)
    cost = got_values.get("signal", {}).get("layout_matching_cost", float("nan"))
    return f"{name},{cost},{got_values.get('is_flipped', False)},{got_values.get('layout_name', '')}\n"


def _append_metadata(output_basepath: str, row: str) -> None:
    path = os.path.join(os.path.dirname(output_basepath), "digitization_metadata.csv")
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("file_path,matching_cost,is_flipped,lead_layout\n")
        f.write(row)


def save_outputs(got_values: dict[str, Any], output_basepath: str, save_mode: str = "all") -> None:
    canonical = canonical_from_got_values(got_values)
    if save_mode in ["all", "timeseries_only"]:
        save_timeseries_csv(canonical, output_basepath)
    if save_mode in ["all", "png_only"]:
        save_png_plot(got_values, canonical, output_basepath)
    save_matching_cost(got_values, output_basepath)


def _layout_hint(file_path: str, config: CN) -> str | None:
    layout_should_include_substring: str | None = None
    if config.DATA.get("layout_should_include_substring") is not None:
        if "limb" in str(file_path):
            layout_should_include_substring = "limb"
        elif "precordial" in str(file_path):
            layout_should_include_substring = "precordial"

    return layout_should_include_substring


def _infer_file(file_path: str, config: CN, inference_wrapper: Any) -> dict[str, Any]:
    image = decode_and_prepare_image(file_path)
    values: dict[str, Any] = inference_wrapper(image, layout_should_include_substring=_layout_hint(file_path, config))
    return values


def _output_basepath(file_path: str, config: CN) -> str:
    rel_path = os.path.relpath(file_path, config.DATA.images_path)
    basepath = os.path.splitext(os.path.join(config.DATA.output_path, rel_path))[0]
    os.makedirs(os.path.dirname(basepath), exist_ok=True)
    return basepath


def process_one_file(file_path: str, config: CN, inference_wrapper: Any, save_mode: str) -> None:
    got_values = _infer_file(file_path, config, inference_wrapper)
    if config.DATA.get("output_path") is not None:
        save_outputs(got_values, _output_basepath(file_path, config), save_mode)


def _run_signal_batch(config: CN, paths: list[str], model: Any, workers: int, pool: ProcessPoolExecutor) -> int:
    from src.model.signal_batch import extract_record, finish_record, prepare_record

    def save_record(path: str, task: Future[Any], detected: list[tuple[str, float, float]]) -> int:
        try:
            result = finish_record(model, task.result(), detected, _layout_hint(path, config))
            save_outputs(result, _output_basepath(path, config), "timeseries_only")
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return 1
        return 0

    # A single writer commits signals and metadata in input order, including
    # when two inputs share an output filename. Bound the number of live maps.
    failed = 0
    pending: deque[Future[int]] = deque()
    with ThreadPoolExecutor(1) as writer:
        for path in tqdm(paths):
            print(f"Processing file: {path}")
            try:
                inputs, detected = prepare_record(model, decode_and_prepare_image(path))
                task = pool.submit(extract_record, model.signal_extractor, model.pixel_size_finder, inputs)
                pending.append(writer.submit(save_record, path, task, detected))
            except Exception as e:
                print(f"Error processing {path}: {e}")
                failed += 1
            if len(pending) >= 2 * workers:
                failed += pending.popleft().result()
        return failed + sum(future.result() for future in pending)


def main(config: CN, seed: int | None = DEFAULT_INFERENCE_SEED, plot_workers: int = 0, signal_workers: int = 0) -> int:
    """Digitize the selected images and return the number of failed files."""
    with ExitStack() as stack:
        pool = None
        if config.DATA.get("save_mode") == "timeseries_only" and signal_workers > 1 and config.DATA.get("output_path"):
            method = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"
            pool = stack.enter_context(
                ProcessPoolExecutor(signal_workers, mp_context=multiprocessing.get_context(method))
            )
            # Forkserver startup imports the CPU stack. Overlap that with loading
            # PyTorch and the network weights in the GPU process.
            starter = stack.enter_context(ThreadPoolExecutor(1))
            starter.submit(pool.submit, os.getpid)
        return _main(config, seed, plot_workers, signal_workers, pool)


def _main(
    config: CN, seed: int | None, plot_workers: int, signal_workers: int, signal_pool: ProcessPoolExecutor | None
) -> int:
    from src.model.inference_wrapper import InferenceWrapper

    inference_wrapper_class = import_class_from_path(config.MODEL.class_path)
    kwargs = dict(config.MODEL.KWARGS)
    if inference_wrapper_class is InferenceWrapper:
        kwargs["seed"] = seed
    else:
        seed_inference(seed)
    inference_wrapper = inference_wrapper_class(**kwargs)
    save_mode: str = getattr(config.DATA, "save_mode", "all")
    file_paths: list[str] = get_candidate_file_paths(config)

    include_list = config.DATA.get("path_should_include", [])

    if config.DATA.get("output_path") is not None:
        clear_and_prepare_output_dir(config)

    paths = [p for p in file_paths if not include_list or any(pattern in p for pattern in include_list)]
    if signal_pool is not None and len(paths) > 1:
        from src.model.signal_batch import supports_signal_batch

        if supports_signal_batch(inference_wrapper):
            return _run_signal_batch(config, paths, inference_wrapper, min(signal_workers, len(paths)), signal_pool)
    plot_workers = min(plot_workers, len(paths))
    pending: deque[tuple[str, str, Future[None], str]] = deque()
    failed = 0

    def finish_oldest() -> None:
        nonlocal failed
        path, base, future, row = pending.popleft()
        try:
            future.result()
            _append_metadata(base, row)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            failed += 1

    with ExitStack() as stack:
        pool = None
        if plot_workers > 1 and save_mode in ("all", "png_only") and config.DATA.get("output_path"):
            pool = stack.enter_context(
                ProcessPoolExecutor(max_workers=plot_workers, mp_context=multiprocessing.get_context("spawn"))
            )
        for file_path in tqdm(paths):
            print(f"Processing file: {file_path}")
            try:
                if pool is None:
                    process_one_file(file_path, config, inference_wrapper, save_mode)
                    continue
                values = _infer_file(file_path, config, inference_wrapper)
                base = _output_basepath(file_path, config)
                # Bound memory and preserve overwrite order for equal filename stems.
                while len(pending) >= plot_workers or any(job[1] == base for job in pending):
                    finish_oldest()
                canonical = canonical_from_got_values(values)
                if save_mode == "all":
                    save_timeseries_csv(canonical, base)
                future = pool.submit(_prepare_plot(values, canonical).save, base)
                pending.append((file_path, base, future, _metadata_row(values, base)))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                failed += 1
        while pending:
            finish_oldest()
    return failed


if __name__ == "__main__":
    # Matplotlib allocates large, short-lived NumPy raster buffers. Requesting
    # transparent huge pages for them can stall concurrent writers in Linux
    # memory compaction. Set this before importing NumPy in the CLI and workers.
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    parser = argparse.ArgumentParser(description="Digitize ECGs and save output.")
    parser.add_argument(
        "--plot-workers",
        type=int,
        default=min(6, os.cpu_count() or 1),
        help="CPU processes for PNG export (default: up to 6). Use 0 for in-process export.",
    )
    parser.add_argument(
        "--signal-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="CPU processes for signal-only tracing (default: up to 4). Use 0 for serial inference.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_INFERENCE_SEED, help="Inference seed (default: 42).")
    parser.add_argument(
        "--config",
        type=str,
        help="Config file name or path (searched in . and src/config/)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override config options like A.B.C=123 or DATA.save_mode='png_only' (spaces require quotes).",
    )
    args = parser.parse_args()

    config_path = find_config_path(args.config)
    cfg = get_cfg(config_path)

    if args.overrides:
        kv_list: list[str] = []
        for ov in args.overrides:
            if "=" not in ov:
                raise ValueError(f"Malformed override '{ov}'. Use KEY=VALUE (e.g., MODEL.device='cuda:0').")
            k, v = ov.split("=", 1)

            # Check if key exists in config before merging
            node = cfg
            key_parts = k.split(".")
            exists = True
            for part in key_parts[:-1]:
                if part in node:
                    node = node[part]
                else:
                    exists = False
                    break
            if not exists or key_parts[-1] not in node:
                warnings.warn(f"Config key '{k}' not found in loaded config. Override skipped.")
                continue

            kv_list.extend([k, v])

        if kv_list:
            cfg.merge_from_list(kv_list)

    if (
        cfg.DATA.get("save_mode") == "timeseries_only"
        and args.signal_workers > 1
        and "forkserver" in multiprocessing.get_all_start_methods()
    ):
        multiprocessing.set_forkserver_preload(["src.model.signal_batch"])
    failed = main(cfg, seed=args.seed, plot_workers=args.plot_workers, signal_workers=args.signal_workers)
    raise SystemExit(int(failed > 0))
