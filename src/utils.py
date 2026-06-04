import importlib
import math
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.optim.optimizer
from ray.tune import Stopper
from torch import nn
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN


class EarlyStopper(Stopper):
    def __init__(self, metric: str, patience: int, delta: float = 0):
        """Stops the training if the metric does not decrease by at least delta for patience epochs."""
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.delta = delta
        self.counter: Dict[str, int] = defaultdict(lambda: 0)
        self.best_score: Dict[str, float] = defaultdict(lambda: math.inf)

    def __call__(self, trial_id: str, result: Dict[str, Any]) -> bool:
        score = result[self.metric]

        if score >= self.best_score[trial_id] - self.delta:
            self.counter[trial_id] += 1
        else:
            self.counter[trial_id] = 0

        self.best_score[trial_id] = min(self.best_score[trial_id], score)

        return self.patience <= self.counter[trial_id]

    def stop_all(self) -> bool:
        return False


def find_config_path(filename: str) -> str:
    """Tries to find the config file in . or src/config.

    Args:
        filename: Config file name or path.

    Returns:
        Absolute path to config.
    """
    if os.path.isfile(filename):
        return filename
    # Try in src/config/
    try_path = os.path.join("src", "config", filename)
    if os.path.isfile(try_path):
        return try_path
    raise FileNotFoundError(f"Could not find config file: {filename}")


def get_data_loaders(
    dataset_config: Dict[Any, Any], dataloader_config: Dict[Any, Any]
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    dataset_model = import_class_from_path(dataset_config["class_path"])

    dataset_kwargs = dataset_config["KWARGS"]

    def get_transform(config: Dict[Any, Any]) -> Any:
        if "TRANSFORM" not in config:
            return None
        transform_config = config["TRANSFORM"]
        transform_class = import_class_from_path(transform_config["class_path"])
        return transform_class(**transform_config.get("KWARGS", {}))

    transform_train = get_transform(dataset_config.get("TRAIN", {}))
    transform_val = get_transform(dataset_config.get("VAL", {}))
    transform_test = get_transform(dataset_config.get("TEST", {}))

    train_dataset = dataset_model(**{**dataset_kwargs, **dataset_config["TRAIN"]["KWARGS"]}, transform=transform_train)
    val_dataset = dataset_model(**{**dataset_kwargs, **dataset_config["VAL"]["KWARGS"]}, transform=transform_val)
    test_dataset = dataset_model(**{**dataset_kwargs, **dataset_config["TEST"]["KWARGS"]}, transform=transform_test)

    train_dataloader = DataLoader(train_dataset, **dataloader_config)
    val_dataloader = DataLoader(val_dataset, **dataloader_config)
    test_dataloader = DataLoader(test_dataset, **dataloader_config)

    return train_dataloader, val_dataloader, test_dataloader


def import_class_from_path(path: str) -> Any:
    module = importlib.import_module(".".join(path.split(".")[:-1]))
    return getattr(module, path.split(".")[-1])


def load_model(config: CN) -> nn.Module:
    model_class = import_class_from_path(config.MODEL.class_path)
    return model_class(**config.MODEL.KWARGS)  # type: ignore


def merge_ray_config_with_config(config: CN, ray_config: CN) -> CN:
    config = deepcopy(config)
    ray_config = deepcopy(ray_config)
    model_kwargs = deepcopy(config)
    for k, v in ray_config.items():
        if k in model_kwargs:
            config.MODEL.KWARGS[k] = v
    return config


class CosineToConstantLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        eta_min_divisor: Optional[float] = None,
    ):  # noqa: D107
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_min_divisor = eta_min_divisor
        super().__init__(optimizer, -1)

    def get_lr(self) -> List[Any]:
        """Retrieve the learning rate of each parameter group."""
        torch.optim.lr_scheduler._warn_get_lr_called_within_step(self)

        def _get_lr(base_lr: float) -> float:
            min_lr = self.eta_min if self.eta_min_divisor is None else base_lr / self.eta_min_divisor
            if self._step_count - 1 >= self.T_max:
                return min_lr
            return min_lr + (base_lr - min_lr) * (1 + (math.cos(math.pi * (self._step_count - 1) / self.T_max))) / 2

        return [_get_lr(float(base_lr)) for base_lr in self.base_lrs]


class SavePlotsEveryN:
    def __init__(self, plot_path: str, n: int, split_name: str) -> None:
        self.n = n
        self.plot_path = plot_path
        self.counter = 0
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        assert isinstance(n, int) and n > 0, "n must be a positive integer"

    def __call__(self, pred: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        if (self.counter % self.n == 0) and (self.counter > 0):
            pred = torch.softmax(pred, dim=1).permute(0, 2, 3, 1)[0].detach().cpu()
            pred = pred[:, :, 0:3]
            target_rgb = target_rgb.permute(0, 2, 3, 1)[0].detach().cpu()

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(pred)
            axs[1].imshow(target_rgb)
            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=300)
            plt.close()

        self.counter += 1

        return torch.Tensor([0])

    @property
    def __name__(self) -> str:
        return f"SavePlotsEveryN(n={self.n})"
