"""Simple supervised trainer built around direct dataloaders."""

from __future__ import annotations

import random
import re
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import SpinnerColumn, TextColumn, TimeElapsedColumn
from torch import nn
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

Scalar = float | int | torch.Tensor
MetricLike = Callable[[Any, Any], Scalar]


@dataclass(frozen=True)
class TrainerConfig:
    """Epoch-level training options consumed by `Trainer.step()` and `Trainer.test()`."""

    max_epochs: int = 1
    train_max_batches: int | None = None
    val_max_batches: int | None = None
    test_max_batches: int | None = None
    checkpoint_dir: str | Path | None = None
    show_progress: bool | None = None

    def __post_init__(self) -> None:
        _validate_positive("max_epochs", self.max_epochs)
        for name, value in (
            ("train_max_batches", self.train_max_batches),
            ("val_max_batches", self.val_max_batches),
            ("test_max_batches", self.test_max_batches),
        ):
            _validate_optional_positive(name, value)


@dataclass
class TrainerState:
    """Mutable training progress tracked across checkpoint save and restore."""

    epoch: int = 0
    global_step: int = 0


class MAE(nn.Module):
    """Mean absolute error metric for regression outputs."""

    name = "mae"

    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(outputs, target)


def _validate_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_optional_positive(name: str, value: int | None) -> int | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return value


def _move_to_device(value: Any, device: torch.device) -> Any:
    # batch가 dict/list/tuple로 중첩돼도 같은 device로 한 번에 옮긴다.
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _to_float(value: Scalar) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _progress_metrics_desc(stage: str, metrics: Mapping[str, float]) -> str:
    # tqdm.rich는 postfix를 제대로 그리지 않아서 description에 metric을 직접 싣는다.
    summary = " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    return f"{stage} | {summary}"


def _epoch_header(epoch: int, total_epochs: int) -> str:
    return f"epoch {epoch + 1}/{total_epochs}"


def _loader_length(dataloader: Any) -> int | None:
    try:
        return len(dataloader)
    except (TypeError, AttributeError, NotImplementedError):
        return None


def _resolve_progress_total(dataloader: Any, max_batches: int | None) -> int | None:
    # streaming loader는 len()이 없을 수 있으므로 max_batches를 fallback으로 쓴다.
    loader_length = _loader_length(dataloader)
    if loader_length is None:
        return max_batches
    if max_batches is None:
        return loader_length
    return min(loader_length, max_batches)


def _indeterminate_progress_columns() -> tuple[Any, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed:>4.0f} batches"),
        TimeElapsedColumn(),
    )


def _create_progress(*, desc: str, total: int | None, disable: bool, leave: bool) -> tqdm:
    progress_kwargs: dict[str, Any] = {
        "total": total,
        "desc": desc,
        "disable": disable,
        "leave": leave,
        "dynamic_ncols": True,
        "unit": "batch",
    }
    if total is None:
        progress_kwargs["progress"] = _indeterminate_progress_columns()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TqdmExperimentalWarning)
        return tqdm(**progress_kwargs)


def _capture_rng_state() -> dict[str, Any]:
    # checkpoint 복구 후에도 batch 순서와 augmentation RNG를 최대한 그대로 재현한다.
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_to_device(value, device)


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


def _validate_optimizer_model_pair(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    if any(id(parameter) not in model_parameter_ids for parameter in _optimizer_parameters(optimizer)):
        raise ValueError("optimizer must be constructed from the parameters of the supplied model")


def _infer_model_device(model: nn.Module) -> torch.device:
    devices = {tensor.device for tensor in list(model.parameters()) + list(model.buffers())}
    if not devices:
        return torch.device("cpu")
    if len(devices) != 1:
        raise ValueError("model parameters and buffers must live on a single device")
    device = next(iter(devices))
    if device.type == "meta":
        raise ValueError("model must be materialized on a real device before passing it to Trainer")
    return device


def _forward_model(model: nn.Module, inputs: Any) -> Any:
    # Keep the trainer compatible with the common supervised patterns:
    # `model(x)`, `model(*inputs)`, and `model(**inputs)`.
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


def _set_loader_epoch(loader: Any, epoch: int) -> None:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "set_epoch"):
        # train dataset은 epoch별 shuffle 상태를 dataset 내부에서 관리한다.
        dataset.set_epoch(epoch)


def _extract_inputs_target(batch: Mapping[str, Any]) -> tuple[Any, Any]:
    if "inputs" not in batch or "target" not in batch:
        raise ValueError("trainer batches must contain 'inputs' and 'target'")
    return batch["inputs"], batch["target"]


def _snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).replace("-", "_").lower()


def _metric_name(metric: MetricLike) -> str:
    explicit_name = getattr(metric, "name", None)
    if isinstance(explicit_name, str) and explicit_name:
        return explicit_name
    function_name = getattr(metric, "__name__", None)
    if isinstance(function_name, str) and function_name:
        if function_name == "<lambda>":
            raise ValueError("lambda metrics are not supported; pass metric objects such as MAE()")
        return _snake_case(function_name)
    return _snake_case(metric.__class__.__name__)


def _prepare_metrics(metrics: Sequence[MetricLike] | None) -> list[tuple[str, MetricLike]]:
    if metrics is None:
        return []
    if isinstance(metrics, Mapping):
        raise TypeError("metrics must be a sequence of metric objects, not a mapping")

    prepared: list[tuple[str, MetricLike]] = []
    seen_names: set[str] = set()
    for metric in metrics:
        if not callable(metric):
            raise TypeError("each metric must be callable")
        name = _metric_name(metric)
        if name in seen_names:
            raise ValueError(f"duplicate metric name: {name}")
        seen_names.add(name)
        prepared.append((name, metric))
    return prepared


class Trainer:
    """Supervised trainer that owns the common train/val/test loop."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[Any, Any], torch.Tensor],
        metrics: Sequence[MetricLike] | None,
        config: TrainerConfig,
        train_loader: Any,
        val_loader: Any | None = None,
        test_loader: Any | None = None,
    ) -> None:
        """Bind the model, optimization objects, direct dataloaders, and metric objects."""

        _validate_optimizer_model_pair(model, optimizer)
        self.model = model
        self.device = _infer_model_device(model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = _prepare_metrics(metrics)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.state = TrainerState()
        self.checkpoint_dir = (
            Path(config.checkpoint_dir) if config.checkpoint_dir is not None else None
        )
        self.show_progress = True if config.show_progress is None else config.show_progress
        _move_optimizer_state_to_device(self.optimizer, self.device)
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _run_stage(
        self,
        stage: str,
        dataloader: Any,
        *,
        max_batches: int | None,
        training: bool,
        epoch: int,
    ) -> dict[str, float]:
        metric_totals: dict[str, float] = {}
        batch_count = 0

        if training:
            # val/test는 고정 순서를 유지해야 하므로 epoch 전달은 train에만 한다.
            _set_loader_epoch(dataloader, epoch)
        self.model.train(training)

        progress = _create_progress(
            total=_resolve_progress_total(dataloader, max_batches),
            desc=stage,
            disable=not self.show_progress,
            leave=True,
        )
        try:
            progress.set_description_str(f"{stage} | loading first batch...")
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                moved_batch = _move_to_device(batch, self.device)
                inputs, target = _extract_inputs_target(moved_batch)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    outputs = _forward_model(self.model, inputs)
                    loss = self.criterion(outputs, target)
                    if training:
                        loss.backward()
                        self.optimizer.step()
                        self.state.global_step += 1

                metric_totals["loss"] = metric_totals.get("loss", 0.0) + _to_float(loss)
                with torch.no_grad():
                    for name, metric in self.metrics:
                        metric_value = metric(outputs, target)
                        metric_totals[name] = metric_totals.get(name, 0.0) + _to_float(metric_value)

                batch_count += 1
                averages = {name: total / batch_count for name, total in metric_totals.items()}
                progress.set_description_str(
                    _progress_metrics_desc(stage, averages),
                    refresh=False,
                )
                progress.update(1)
        finally:
            progress.close()

        if batch_count == 0:
            return {}
        return {name: total / batch_count for name, total in metric_totals.items()}

    def step(
        self,
        *,
        max_epochs: int | None = None,
        train_max_batches: int | None = None,
        val_max_batches: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run epochs and yield one history record per completed epoch."""

        target_epochs = (
            self.config.max_epochs
            if max_epochs is None
            else _validate_positive("max_epochs", max_epochs)
        )
        active_train_max_batches = (
            self.config.train_max_batches
            if train_max_batches is None
            else _validate_optional_positive("train_max_batches", train_max_batches)
        )
        active_val_max_batches = (
            self.config.val_max_batches
            if val_max_batches is None
            else _validate_optional_positive("val_max_batches", val_max_batches)
        )

        while self.state.epoch < target_epochs:
            epoch = self.state.epoch
            if self.show_progress:
                # epoch 헤더를 먼저 출력해 stage progress가 어느 epoch 소속인지 바로 보이게 한다.
                print(_epoch_header(epoch, target_epochs), flush=True)
            train_metrics = self._run_stage(
                "train",
                self.train_loader,
                max_batches=active_train_max_batches,
                training=True,
                epoch=epoch,
            )
            val_metrics = (
                self._run_stage(
                    "val",
                    self.val_loader,
                    max_batches=active_val_max_batches,
                    training=False,
                    epoch=epoch,
                )
                if self.val_loader is not None
                else {}
            )

            self.state.epoch += 1
            self.save_checkpoint("last.pt")
            self.save_checkpoint(f"epoch-{self.state.epoch:04d}.pt")

            yield {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "train": train_metrics,
                "val": val_metrics,
            }

    def test(self, *, max_batches: int | None = None) -> dict[str, float]:
        """Evaluate the configured test loader and return averaged metrics."""

        if self.test_loader is None:
            raise ValueError("test_loader is required to run test()")
        active_max_batches = (
            self.config.test_max_batches
            if max_batches is None
            else _validate_optional_positive("max_batches", max_batches)
        )
        return self._run_stage(
            "test",
            self.test_loader,
            max_batches=active_max_batches,
            training=False,
            epoch=self.state.epoch,
        )

    def save_checkpoint(self, filename: str | Path) -> Path | None:
        """Save model, optimizer, RNG, and trainer state when checkpointing is enabled."""

        if self.checkpoint_dir is None:
            return None
        path = self.checkpoint_dir / filename
        checkpoint = {
            "state": {"epoch": self.state.epoch, "global_step": self.state.global_step},
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": _capture_rng_state(),
        }
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model, optimizer, RNG, and trainer state from a checkpoint."""

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state_to_device(self.optimizer, self.device)
        state = checkpoint["state"]
        self.state = TrainerState(epoch=state["epoch"], global_step=state["global_step"])
        _restore_rng_state(checkpoint["rng"])
