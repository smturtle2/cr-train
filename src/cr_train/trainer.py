from __future__ import annotations

import random
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from tqdm import TqdmExperimentalWarning
from torch import nn
from torch.nn import Parameter
from tqdm.rich import tqdm

from .data import Stage

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

Scalar = float | int | torch.Tensor


@dataclass
class StepResult:
    loss: torch.Tensor | None = None
    metrics: dict[str, Scalar] = field(default_factory=dict)


class StepFn(Protocol):
    def __call__(self, model: nn.Module, batch: Mapping[str, Any], stage: Stage) -> StepResult: ...


SchedulerStepFn = Callable[[Any, Mapping[str, float]], None]


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0


def _move_to_device(value: Any, device: torch.device) -> Any:
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


def _progress_desc(stage: Stage, epoch: int) -> str:
    if stage == "test":
        return "test"
    return f"epoch {epoch + 1} {stage}"


def _capture_rng_state() -> dict[str, Any]:
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
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[Parameter]:
    params: list[Parameter] = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    return params


def _remap_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
    parameter_map: Mapping[Parameter, Parameter],
) -> None:
    for group in optimizer.param_groups:
        group["params"] = [parameter_map.get(param, param) for param in group["params"]]

    default_factory = getattr(optimizer.state, "default_factory", None)
    remapped_state = type(optimizer.state)(default_factory) if default_factory is not None else type(optimizer.state)()
    for param, state in optimizer.state.items():
        remapped_state[parameter_map.get(param, param)] = state
    optimizer.state = remapped_state


def _align_model_and_optimizer_to_device(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> nn.Module:
    current_parameters = list(model.parameters())
    optimizer_parameters = _optimizer_parameters(optimizer)
    current_parameter_ids = {id(parameter) for parameter in current_parameters}

    if any(id(parameter) not in current_parameter_ids for parameter in optimizer_parameters):
        raise ValueError("optimizer must be constructed from the parameters of the supplied model")

    moved_model = model.to(device)
    moved_parameters = list(moved_model.parameters())
    if len(current_parameters) != len(moved_parameters):
        raise RuntimeError("model parameter count changed while moving to device")

    parameter_map = {
        current_parameter: moved_parameter
        for current_parameter, moved_parameter in zip(current_parameters, moved_parameters, strict=True)
    }
    _remap_optimizer_parameters(optimizer, parameter_map)
    _move_optimizer_state_to_device(optimizer, device)
    return moved_model


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        datamodule: Any,
        step_fn: StepFn,
        scheduler: Any | None = None,
        scheduler_step_fn: SchedulerStepFn | None = None,
        device: torch.device | str | None = None,
        checkpoint_dir: str | Path | None = None,
        show_progress: bool | None = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = _align_model_and_optimizer_to_device(model, optimizer, self.device)
        self.datamodule = datamodule
        self.step_fn = step_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_fn = scheduler_step_fn
        self.state = TrainerState()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.show_progress = sys.stderr.isatty() if show_progress is None else show_progress
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _run_stage(
        self,
        stage: Stage,
        dataloader: Any,
        *,
        max_batches: int | None,
        training: bool,
        epoch: int,
    ) -> dict[str, float]:
        metric_totals: dict[str, float] = {}
        batch_count = 0

        self.model.train(training)
        if not training:
            self.model.eval()

        progress = tqdm(
            total=max_batches,
            desc=_progress_desc(stage, epoch),
            disable=not self.show_progress,
            leave=False,
            dynamic_ncols=True,
            unit="batch",
        )
        try:
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                moved_batch = _move_to_device(batch, self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    result = self.step_fn(self.model, moved_batch, stage)
                    if training:
                        if result.loss is None:
                            raise ValueError("training step_fn must return a loss tensor")
                        result.loss.backward()
                        self.optimizer.step()
                        self.state.global_step += 1

                if result.loss is not None:
                    metric_totals["loss"] = metric_totals.get("loss", 0.0) + _to_float(result.loss)

                for name, value in result.metrics.items():
                    metric_totals[name] = metric_totals.get(name, 0.0) + _to_float(value)

                batch_count += 1
                progress.update(1)

                averages = {name: total / batch_count for name, total in metric_totals.items()}
                progress.set_postfix({key: f"{value:.4f}" for key, value in averages.items()})
        finally:
            progress.close()

        if batch_count == 0:
            return {}

        return {name: total / batch_count for name, total in metric_totals.items()}

    def fit(
        self,
        *,
        max_epochs: int,
        train_max_batches: int | None = None,
        val_max_batches: int | None = None,
    ) -> list[dict[str, dict[str, float]]]:
        history: list[dict[str, dict[str, float]]] = []

        while self.state.epoch < max_epochs:
            epoch = self.state.epoch
            train_loader = self.datamodule.train_dataloader(epoch=epoch)
            train_metrics = self._run_stage(
                "train",
                train_loader,
                max_batches=train_max_batches,
                training=True,
                epoch=epoch,
            )

            val_loader = self.datamodule.val_dataloader(epoch=epoch)
            val_metrics = self._run_stage(
                "val",
                val_loader,
                max_batches=val_max_batches,
                training=False,
                epoch=epoch,
            )

            epoch_metrics = {"train": train_metrics, "val": val_metrics}
            history.append(epoch_metrics)

            if self.scheduler is not None and self.scheduler_step_fn is not None:
                self.scheduler_step_fn(self.scheduler, {**train_metrics, **{f"val/{k}": v for k, v in val_metrics.items()}})

            self.state.epoch += 1
            self.save_checkpoint("last.pt")
            self.save_checkpoint(f"epoch-{self.state.epoch:04d}.pt")

        return history

    def test(self, *, test_max_batches: int | None = None) -> dict[str, float]:
        test_loader = self.datamodule.test_dataloader(epoch=self.state.epoch)
        return self._run_stage(
            "test",
            test_loader,
            max_batches=test_max_batches,
            training=False,
            epoch=self.state.epoch,
        )

    def save_checkpoint(self, filename: str | Path) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        path = self.checkpoint_dir / filename
        checkpoint = {
            "state": {"epoch": self.state.epoch, "global_step": self.state.global_step},
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": _capture_rng_state(),
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            checkpoint["scheduler"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state_to_device(self.optimizer, self.device)
        if self.scheduler is not None and "scheduler" in checkpoint and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        state = checkpoint["state"]
        self.state = TrainerState(epoch=state["epoch"], global_step=state["global_step"])
        _restore_rng_state(checkpoint["rng"])
