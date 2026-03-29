from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn

from .data import Stage

Scalar = float | int | torch.Tensor


@dataclass
class StepResult:
    loss: torch.Tensor | None = None
    metrics: dict[str, Scalar] = field(default_factory=dict)


class StepFn(Protocol):
    def __call__(self, model: nn.Module, batch: Mapping[str, Any], stage: Stage) -> StepResult: ...


OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer]
SchedulerFactory = Callable[[torch.optim.Optimizer], Any]
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


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        datamodule: Any,
        step_fn: StepFn,
        optimizer_factory: OptimizerFactory,
        scheduler_factory: SchedulerFactory | None = None,
        scheduler_step_fn: SchedulerStepFn | None = None,
        device: torch.device | str | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.datamodule = datamodule
        self.step_fn = step_fn
        self.optimizer = optimizer_factory(self.model)
        self.scheduler = scheduler_factory(self.optimizer) if scheduler_factory is not None else None
        self.scheduler_step_fn = scheduler_step_fn
        self.state = TrainerState()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _run_stage(
        self,
        stage: Stage,
        dataloader: Any,
        *,
        max_batches: int | None,
        training: bool,
    ) -> dict[str, float]:
        metric_totals: dict[str, float] = {}
        batch_count = 0

        self.model.train(training)
        if not training:
            self.model.eval()

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
            )

            val_loader = self.datamodule.val_dataloader(epoch=epoch)
            val_metrics = self._run_stage(
                "val",
                val_loader,
                max_batches=val_max_batches,
                training=False,
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
        return self._run_stage("test", test_loader, max_batches=test_max_batches, training=False)

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
