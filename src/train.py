import argparse
import multiprocessing
import os
import tempfile
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import torch._dynamo
from ray.train import Checkpoint
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from torch import nn
from tqdm import tqdm
from yacs.config import CfgNode as CN

from src.config.default import get_cfg, merge_cfg
from src.utils import (
    find_config_path,
    get_data_loaders,
    import_class_from_path,
    load_model,
)


def run_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: torch.utils.data.DataLoader[Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    mixed_precision_scaler: Optional[torch.amp.grad_scaler.GradScaler] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    eval: bool = False,
    epoch: Optional[int] = None,
    max_epochs: Optional[int] = None,
    device: str | torch.device = "cpu",
    return_epoch_data: bool = True,
    metric_instances: Optional[List[nn.Module]] = None,
    disable_tqdm: bool = False,
) -> Tuple[float, int, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, float]]:
    if not eval and optimizer is None:
        raise ValueError("Optimizer must be provided for training (eval=False).")

    if len(dataloader) == 0:
        raise ValueError("Dataloader must not be empty.")

    if not isinstance(device, torch.device):
        device = torch.device(device)

    if eval:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    num_cases = 0
    predictions = []
    targets = []
    progress_bar = tqdm(dataloader) if not disable_tqdm else dataloader

    metrics = metric_instances or []
    stored_metrics: Dict[str, List[torch.Tensor]] = {metric.__name__: [] for metric in metrics}  # type: ignore

    for predictors, target in progress_bar:
        with torch.set_grad_enabled(not eval):
            if optimizer:
                optimizer.zero_grad()
            predictors = predictors.to(device)
            target = target.to(device)
            with torch.autocast(device_type=str(device), enabled=mixed_precision_scaler is not None):
                output = model(predictors)

                if return_epoch_data:
                    predictions.append(output.detach().cpu())
                    targets.append(target.detach().cpu())

                loss = criterion(output, target)

                with torch.no_grad():
                    for metric in metrics:
                        metric_value = metric(output, target)
                        stored_metrics[metric.__name__].append(metric_value.detach().cpu().float().numpy())  # type: ignore

                if eval:
                    pass
                elif mixed_precision_scaler:
                    mixed_precision_scaler.scale(loss).backward()
                    mixed_precision_scaler.step(optimizer)  # type: ignore
                    mixed_precision_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()  # type: ignore

                if lr_scheduler and not eval:
                    lr_scheduler.step()

            total_loss += loss.item()
            num_cases += predictors.shape[0]

            if epoch and not disable_tqdm:
                max_epochs_str = f"/{max_epochs}" if max_epochs else ""
                progress_bar.set_description(f"Epoch {epoch + 1}{max_epochs_str}, Loss: {total_loss / num_cases:.4f}")  # type: ignore

    calculated_metrics: Dict[str, float] = {key: float(np.mean(value)) for key, value in stored_metrics.items()}

    return total_loss, num_cases, predictors, predictions, targets, calculated_metrics


def train(
    model: nn.Module,
    criterion: nn.Module,
    train_dataloader: torch.utils.data.DataLoader[Any],
    val_dataloader: torch.utils.data.DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    mixed_precision_scaler: Optional[torch.amp.grad_scaler.GradScaler],
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    use_ray: bool,
    epochs: int,
    metrics: Dict[Any, Any],
    device: str | torch.device = "cpu",
) -> None:
    if not isinstance(device, torch.device):
        print(f"Using device: {device}")
        device = torch.device(device)

    # print(f"Training model:\n {model}")
    print(f"Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.to(device)

    running_train_losses = []
    running_val_losses = []
    num_epochs = epochs

    metrics_per_split = {}
    val_split_prefix = "val"
    train_split_prefix = "train"

    for split_name in [train_split_prefix, val_split_prefix]:
        curr_metrics = []
        for metric in metrics:
            if metric.get("fit", False):
                metric_class = import_class_from_path(metric["class_path"])(split_name=split_name, **metric["KWARGS"])
                metric_class.fit(train_dataloader.dataset.data)  # type: ignore
            else:
                metric_class = import_class_from_path(metric["class_path"])(split_name=split_name, **metric["KWARGS"])
            curr_metrics.append(metric_class)
        metrics_per_split[split_name] = curr_metrics

    best_val_metrics = {metric.__name__: -np.inf for metric in metrics_per_split[val_split_prefix]}
    lowest_val_loss = np.inf

    epoch_bar = tqdm(range(num_epochs), desc="Training epochs") if use_ray else range(num_epochs)

    for epoch in epoch_bar:
        train_loss, train_cases, _, _, _, train_metrics = run_epoch(
            model,
            criterion,
            train_dataloader,
            optimizer,
            mixed_precision_scaler=mixed_precision_scaler,
            lr_scheduler=lr_scheduler,
            eval=False,
            epoch=epoch,
            max_epochs=num_epochs,
            device=device,
            return_epoch_data=False,
            metric_instances=metrics_per_split[train_split_prefix],
            disable_tqdm=use_ray,
        )
        running_train_losses.append(train_loss / train_cases)

        val_loss, val_cases, _, _, _, val_metrics = run_epoch(
            model,
            criterion,
            val_dataloader,
            None,
            mixed_precision_scaler=mixed_precision_scaler,
            eval=True,
            epoch=epoch,
            max_epochs=num_epochs,
            device=device,
            return_epoch_data=False,
            metric_instances=metrics_per_split[val_split_prefix],
            disable_tqdm=use_ray,
        )

        running_val_losses.append(val_loss / val_cases)

        calculated_metrics = train_metrics | val_metrics
        curr_val_loss = val_loss / val_cases
        is_new_best_metric = curr_val_loss < lowest_val_loss
        lowest_val_loss = min(curr_val_loss, lowest_val_loss)

        for name, value in val_metrics.items():
            if value < best_val_metrics[name]:
                best_val_metrics[name] = value
                is_new_best_metric = True

        results = {"train_loss": train_loss / train_cases, "val_loss": curr_val_loss} | calculated_metrics

        if not use_ray:
            print(f"Epoch {epoch + 1}/{num_epochs}: {results}")
            if os.path.exists("sandbox"):
                torch.save(model.state_dict(), f"sandbox/weights_{epoch + 1}.pt")
        elif is_new_best_metric:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                ray.train.report(
                    results,
                    checkpoint=checkpoint,
                )
        else:
            ray.train.report(results)

    print(f"Train losses: {running_train_losses}")
    print(f"Val losses: {running_val_losses}")
    print(f"Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


def load_and_train(ray_config: CN, config: CN) -> None:
    if ray_config:
        config = merge_cfg(config, CN(ray_config))
        with open(os.path.join(ray.train.get_context().get_trial_dir(), "config.yml"), "w") as f:
            f.write(config.dump())

    model = load_model(config)

    optimizer = import_class_from_path(config.TRAIN.OPTIMIZER.class_path)(
        model.parameters(), **config.TRAIN.OPTIMIZER.KWARGS
    )
    mixed_precision_scaler = (
        import_class_from_path(config.TRAIN.MIXED_PRECISION_SCALER.class_path)(
            model.parameters(), **config.TRAIN.MIXED_PRECISION_SCALER.KWARGS
        )
        if hasattr(config.TRAIN, "MIXED_PRECISION_SCALER")
        else None
    )
    lr_scheduler = (
        import_class_from_path(config.TRAIN.LR_SCHEDULER.class_path)(optimizer, **config.TRAIN.LR_SCHEDULER.KWARGS)
        if hasattr(config.TRAIN, "LR_SCHEDULER")
        else None
    )
    criterion_kwargs = config.TRAIN.CRITERION.KWARGS if hasattr(config.TRAIN, "CRITERION") else {}
    for key, value in criterion_kwargs.items():
        if isinstance(value, list):
            criterion_kwargs[key] = torch.tensor(value, dtype=torch.float32).to(config.TRAIN.KWARGS.device)
    criterion = import_class_from_path(config.TRAIN.CRITERION.class_path)(**criterion_kwargs)
    train_dataloader, val_dataloader, test_dataloder = get_data_loaders(config.DATASET, config.DATALOADER.KWARGS)

    train_fn = train

    if config.TRAIN.COMPILE:
        model = torch.compile(model)  # type: ignore
    train_fn(
        model,
        criterion,
        train_dataloader,
        val_dataloader,
        optimizer,
        mixed_precision_scaler=mixed_precision_scaler,
        lr_scheduler=lr_scheduler,
        use_ray=bool(ray_config),
        **config.TRAIN.KWARGS,
    )
    return None


def load_hyperparameter_search(config: CN) -> Any | Dict[str, Any]:
    # TODO: Add support for nested and conditional spaces (https://docs.ray.io/en/latest/tune/faq.html#nested-spaces).
    if "SAMPLE_TYPE" in config:
        return import_class_from_path(config["SAMPLE_TYPE"])(**config["KWARGS"])

    parsed_config = {}
    for key, value in config.items():
        if isinstance(value, dict):
            parsed_config[key] = load_hyperparameter_search(value)
    return parsed_config


def main(config: CN) -> Optional[ExperimentAnalysis]:
    torch.backends.cudnn.benchmark = config.TRAIN.CUDNN_BENCHMARK
    if not config.HYPERPARAMETER_SEARCH.SEARCH_SPACE:
        return load_and_train(None, config)  # type: ignore

    ray_config = load_hyperparameter_search(config.HYPERPARAMETER_SEARCH.SEARCH_SPACE)
    scheduler = (
        import_class_from_path(config.HYPERPARAMETER_SEARCH.SCHEDULER.class_path)(
            **config.HYPERPARAMETER_SEARCH.SCHEDULER.KWARGS
        )
        if "SCHEDULER" in config.HYPERPARAMETER_SEARCH
        else None
    )
    stopper = (
        import_class_from_path(config.TRAIN.STOPPER.class_path)(**config.TRAIN.STOPPER.KWARGS)
        if "STOPPER" in config.TRAIN
        else None
    )

    # Set seed for Ray Tune's random search. If you remove this line, you will
    # get different configurations each time you run the script.
    np.random.seed(42)

    result: ExperimentAnalysis = ray.tune.run(
        partial(load_and_train, config=config),
        resources_per_trial={"cpu": 16, "gpu": 1},
        config=ray_config,
        num_samples=1,
        scheduler=scheduler,
        stop=stopper,
    )
    return result


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # CUDA does not support "fork", which is default on linux.
    parser = argparse.ArgumentParser(description="Train a model with the specified configuration.")
    parser.add_argument(
        "--config",
        type=str,
        default="unet.yml",
        help="Config file name or path (searched in . and src/config/). Default: unet.yml",
    )
    args = parser.parse_args()

    config_path = find_config_path(args.config)
    cfg = get_cfg(config_path)

    main(cfg)
