"""Simple supervised trainer built around direct dataloaders."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch import nn

Scalar = float | int | torch.Tensor
MetricLike = Callable[[Any, Any], Scalar]


@dataclass(frozen=True)
class TrainerConfig:
    """Epoch-level training options consumed by `Trainer.step()` and `Trainer.test()`."""

    max_epochs: int = 1
    train_max_samples: int | None = None
    val_max_samples: int | None = None
    test_max_samples: int | None = None
    checkpoint_dir: str | Path | None = None
    show_progress: bool | None = None

    def __post_init__(self) -> None:
        _validate_positive("max_epochs", self.max_epochs)
        for name, value in (
            ("train_max_samples", self.train_max_samples),
            ("val_max_samples", self.val_max_samples),
            ("test_max_samples", self.test_max_samples),
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


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
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
    except TypeError:
        return None


def _loader_batch_size(dataloader: Any) -> int | None:
    batch_size = getattr(dataloader, "batch_size", None)
    if isinstance(batch_size, int) and batch_size > 0:
        return batch_size
    return None


def _resolve_progress_total(dataloader: Any, max_samples: int | None) -> int | None:
    # tqdm는 batch 진행률을 보여준다. sample cap이 있더라도 batch total을 안전하게 추정할 때만 채운다.
    loader_length = _loader_length(dataloader)
    if loader_length is None:
        return None
    if max_samples is None:
        return loader_length
    batch_size = _loader_batch_size(dataloader)
    if batch_size is None:
        return None
    capped_batches = (max_samples + batch_size - 1) // batch_size
    return min(loader_length, capped_batches)


class _StageProgress:
    """Thin adapter over rich.progress.Progress used by _run_stage."""

    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    def set_description_str(self, desc: str, refresh: bool = True) -> None:
        self._progress.update(self._task_id, description=desc)
        if refresh:
            self._progress.refresh()

    def update(self, advance: int) -> None:
        self._progress.update(self._task_id, advance=advance)

    def refresh(self) -> None:
        self._progress.refresh()

    def close(self) -> None:
        self._progress.stop()


def _determinate_progress_columns() -> tuple[Any, ...]:
    return (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("batch"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def _indeterminate_progress_columns() -> tuple[Any, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed:>4.0f} batches"),
        TimeElapsedColumn(),
    )


def _create_progress(*, desc: str, total: int | None, disable: bool, leave: bool) -> _StageProgress:
    columns = _indeterminate_progress_columns() if total is None else _determinate_progress_columns()
    progress = Progress(*columns, disable=disable, transient=not leave)
    task_id = progress.add_task(desc, total=total)
    progress.start()
    return _StageProgress(progress, task_id)


def _batch_size(target: Any) -> int:
    if isinstance(target, torch.Tensor):
        if target.ndim == 0:
            raise ValueError("trainer target tensors must include a leading batch dimension")
        return int(target.shape[0])
    try:
        return len(target)
    except TypeError as exc:
        raise ValueError("trainer targets must be batched sequences or tensors") from exc


def _slice_batch_value(value: Any, limit: int, *, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != batch_size:
            return value
        return value[:limit]
    if isinstance(value, Mapping):
        return {
            key: _slice_batch_value(item, limit, batch_size=batch_size)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) == batch_size and all(
            not isinstance(item, (torch.Tensor, Mapping, list, tuple)) for item in value
        ):
            return value[:limit]
        return [_slice_batch_value(item, limit, batch_size=batch_size) for item in value]
    if isinstance(value, tuple):
        if len(value) == batch_size and all(
            not isinstance(item, (torch.Tensor, Mapping, list, tuple)) for item in value
        ):
            return value[:limit]
        return tuple(_slice_batch_value(item, limit, batch_size=batch_size) for item in value)
    return value


def _slice_batch(batch: Any, limit: int) -> Any:
    _, target = _extract_inputs_target(batch)
    batch_size = _batch_size(target)
    if limit >= batch_size:
        return batch
    return _slice_batch_value(batch, limit, batch_size=batch_size)


def _stage_samples(dataloader: Any, max_samples: int | None) -> Iterator[Any]:
    iterator = iter(dataloader)
    remaining_samples = max_samples
    try:
        if remaining_samples is None:
            for batch in iterator:
                yield batch
            return
        while remaining_samples > 0:
            batch = next(iterator)
            _, target = _extract_inputs_target(batch)
            batch_sample_count = _batch_size(target)
            if batch_sample_count <= remaining_samples:
                yield batch
                remaining_samples -= batch_sample_count
                continue
            yield _slice_batch(batch, remaining_samples)
            break
    except StopIteration:
        return
    finally:
        # max_samples로 조기 종료 시 prefetch된 미소비 배치와 worker 상태를 즉시 정리한다.
        del iterator


def _ensure_progress_invariant(stage: str, total: int | None, processed_batches: int) -> None:
    if total is None:
        return
    if processed_batches != total:
        raise RuntimeError(
            f"{stage} stage ended after {processed_batches} batches, expected {total} batches"
        )


def _numpy_rng_to_safe(state: tuple[Any, ...]) -> dict[str, Any]:
    """numpy RNG 상태를 torch.save(weights_only=True)가 직렬화할 수 있는 형태로 변환한다."""
    kind, keys, pos, has_gauss, gauss = state
    return {
        "kind": kind,
        "keys": torch.from_numpy(keys.copy()),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "gauss": float(gauss),
    }


def _safe_to_numpy_rng(safe: Mapping[str, Any]) -> tuple[Any, ...]:
    """torch-safe 형태를 numpy가 받아들이는 RNG 상태 tuple로 복원한다."""
    return (
        safe["kind"],
        safe["keys"].numpy(),
        safe["pos"],
        safe["has_gauss"],
        safe["gauss"],
    )


def _capture_rng_state() -> dict[str, Any]:
    # checkpoint 복구 후에도 batch 순서와 augmentation RNG를 최대한 그대로 재현한다.
    # numpy 상태는 ndarray를 포함하므로 torch tensor로 변환해 weights_only=True를 허용한다.
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": _numpy_rng_to_safe(np.random.get_state()),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(_safe_to_numpy_rng(state["numpy"]))
    torch.random.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_value_to_device(value, device)


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


def _forward_model(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> Any:
    return model(*inputs)


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
        scheduler: Any | None = None,
    ) -> None:
        """Bind the model, optimization objects, direct dataloaders, and metric objects."""

        _validate_optimizer_model_pair(model, optimizer)
        self.model = model
        self.device = _infer_model_device(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        max_samples: int | None,
        training: bool,
        epoch: int,
    ) -> dict[str, float]:
        # per-sample 가중 평균을 위해 metric 합산은 batch_size를 곱해서 누적한다.
        metric_totals: dict[str, float] = {}
        sample_count = 0
        batch_count = 0
        progress_total = _resolve_progress_total(dataloader, max_samples)

        if training:
            # val/test는 고정 순서를 유지해야 하므로 epoch 전달은 train에만 한다.
            _set_loader_epoch(dataloader, epoch)
        self.model.train(training)

        progress = _create_progress(
            total=progress_total,
            desc=stage,
            disable=not self.show_progress,
            leave=True,
        )
        try:
            progress.set_description_str(f"{stage} | loading first batch...")
            for batch in _stage_samples(dataloader, max_samples):
                moved_batch = _move_value_to_device(batch, self.device)
                inputs, target = _extract_inputs_target(moved_batch)
                batch_size = target.shape[0]

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    outputs = _forward_model(self.model, inputs)
                    loss = self.criterion(outputs, target)
                    if training:
                        loss.backward()
                        self.optimizer.step()
                        self.state.global_step += 1

                # criterion/metric은 기본적으로 batch 내 평균을 반환하므로
                # batch_size를 곱해 sample 단위 합계로 변환한다.
                metric_totals["loss"] = metric_totals.get("loss", 0.0) + _to_float(loss) * batch_size
                with torch.no_grad():
                    for name, metric in self.metrics:
                        metric_value = metric(outputs, target)
                        metric_totals[name] = metric_totals.get(name, 0.0) + _to_float(metric_value) * batch_size

                sample_count += batch_size
                batch_count += 1
                averages = {name: total / sample_count for name, total in metric_totals.items()}
                progress.set_description_str(
                    _progress_metrics_desc(stage, averages),
                    refresh=False,
                )
                progress.update(1)
            progress.refresh()
            _ensure_progress_invariant(stage, progress_total, batch_count)
        finally:
            progress.refresh()
            progress.close()

        if sample_count == 0:
            return {}
        return {name: total / sample_count for name, total in metric_totals.items()}

    def step(
        self,
        *,
        max_epochs: int | None = None,
        train_max_samples: int | None = None,
        val_max_samples: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run epochs and yield one history record per completed epoch."""

        target_epochs = (
            self.config.max_epochs
            if max_epochs is None
            else _validate_positive("max_epochs", max_epochs)
        )
        active_train_max_samples = (
            self.config.train_max_samples
            if train_max_samples is None
            else _validate_optional_positive("train_max_samples", train_max_samples)
        )
        active_val_max_samples = (
            self.config.val_max_samples
            if val_max_samples is None
            else _validate_optional_positive("val_max_samples", val_max_samples)
        )

        while self.state.epoch < target_epochs:
            epoch = self.state.epoch
            if self.show_progress:
                # epoch 헤더를 먼저 출력해 stage progress가 어느 epoch 소속인지 바로 보이게 한다.
                print(_epoch_header(epoch, target_epochs), flush=True)
            train_metrics = self._run_stage(
                "train",
                self.train_loader,
                max_samples=active_train_max_samples,
                training=True,
                epoch=epoch,
            )
            val_metrics = (
                self._run_stage(
                    "val",
                    self.val_loader,
                    max_samples=active_val_max_samples,
                    training=False,
                    epoch=epoch,
                )
                if self.val_loader is not None
                else {}
            )

            if self.scheduler is not None:
                self.scheduler.step()
            self.state.epoch += 1
            self.save_checkpoint("last.pt")
            self.save_checkpoint(f"epoch-{self.state.epoch:04d}.pt")

            yield {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "train": train_metrics,
                "val": val_metrics,
            }

    def test(self, *, max_samples: int | None = None) -> dict[str, float]:
        """Evaluate the configured test loader and return averaged metrics."""

        if self.test_loader is None:
            raise ValueError("test_loader is required to run test()")
        active_max_samples = (
            self.config.test_max_samples
            if max_samples is None
            else _validate_optional_positive("max_samples", max_samples)
        )
        return self._run_stage(
            "test",
            self.test_loader,
            max_samples=active_max_samples,
            training=False,
            epoch=self.state.epoch,
        )

    def save_checkpoint(self, filename: str | Path) -> Path | None:
        """Save model, optimizer, RNG, and trainer state when checkpointing is enabled."""

        if self.checkpoint_dir is None:
            return None
        path = self.checkpoint_dir / filename
        checkpoint: dict[str, Any] = {
            "state": {"epoch": self.state.epoch, "global_step": self.state.global_step},
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": _capture_rng_state(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model, optimizer, RNG, and trainer state from a checkpoint."""

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state_to_device(self.optimizer, self.device)
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        state = checkpoint["state"]
        self.state = TrainerState(epoch=state["epoch"], global_step=state["global_step"])
        _restore_rng_state(checkpoint["rng"])
