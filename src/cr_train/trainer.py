from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from .data.constants import DATASET_ID
from .data.dataset import (
    PreparedSplitState,
    build_dataloader,
    move_batch_to_device,
    prepare_split_from_state,
    resolve_num_workers,
    resolve_prepared_split_state,
    seed_everything,
)
from .data.runtime import ensure_split_cache, is_distributed, is_primary
from .data.store import resolve_cache_root
from .data.source import run_startup_stage
from .progress import resolve_progress_bar_ncols
from .trainer_reporting import (
    format_config_banner,
    format_startup_message,
    format_test_summary,
    format_train_epoch_row,
    format_val_epoch_row,
    serialize_value,
    should_print_startup,
)
from .trainer_runtime import (
    MetricAccumulator,
    compute_loss,
    compute_metric_values,
    finalize_summary,
    prime_iterator,
    update_progress_bar,
)


LossFn = Callable[[Any, Mapping[str, Any]], torch.Tensor | float | int]
MetricFn = Callable[[Any, Mapping[str, Any]], torch.Tensor | float | int]
_MULTIPROCESSING_CONTEXT_CHOICES = {"fork", "spawn", "forkserver"}
_SCHEDULER_TIMING_CHOICES = {
    "after_validation",
    "before_optimizer_step",
    "after_optimizer_step",
}
SchedulerTiming = Literal[
    "after_validation",
    "before_optimizer_step",
    "after_optimizer_step",
]


@dataclass(slots=True)
class _PreparedSplitCacheEntry:
    split: str
    max_samples: int | None
    state: PreparedSplitState


@dataclass(frozen=True, slots=True)
class _ResolvedSchedulerMonitor:
    path: str
    metric_name: str | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedSchedulerConfig:
    timing: SchedulerTiming
    monitor: _ResolvedSchedulerMonitor | None = None


class Trainer:
    """SEN12MS-CR trainer.

    `seed` fixes deterministic logical block selection and epoch-wise train sample order.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: LossFn,
        metrics: Mapping[str, MetricFn] | None = None,
        *,
        scheduler: LRScheduler | None = None,
        scheduler_timing: SchedulerTiming = "after_validation",
        scheduler_monitor: str | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        batch_size: int = 4,
        accum_steps: int = 1,
        epochs: int = 1,
        seed: int = 42,
        output_dir: str | Path = "runs/default",
        cache_dir: str | Path | None = None,
        num_workers: int | str = "auto",
        multiprocessing_context: str | None = None,
        train_crop_size: int | None = 128,
        train_random_flip: bool = True,
        train_random_rot90: bool = True,
        grad_clip_norm: float | None = 1.0,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer instance")
        if not callable(loss):
            raise TypeError("loss must be callable")
        if scheduler is not None and not isinstance(scheduler, LRScheduler):
            raise TypeError("scheduler must be a torch.optim.lr_scheduler.LRScheduler instance")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if accum_steps <= 0:
            raise ValueError("accum_steps must be greater than zero")
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        if train_crop_size is not None and train_crop_size <= 0:
            raise ValueError("train_crop_size must be greater than zero when provided")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be greater than zero when provided")

        self._validate_max_samples("max_train_samples", max_train_samples)
        self._validate_max_samples("max_val_samples", max_val_samples)
        self._validate_max_samples("max_test_samples", max_test_samples)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss
        self.metric_fns = dict(metrics or {})
        for name, metric_fn in self.metric_fns.items():
            if not callable(metric_fn):
                raise TypeError(f"metric '{name}' must be callable")
        self._validate_optimizer_matches_model()
        self._validate_scheduler_matches_optimizer()
        self._scheduler_config = self._resolve_scheduler_config(
            scheduler_timing=scheduler_timing,
            scheduler_monitor=scheduler_monitor,
        )
        self.scheduler_timing = self._scheduler_config.timing
        self.scheduler_monitor = (
            self._scheduler_config.monitor.path
            if self._scheduler_config.monitor is not None
            else None
        )

        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.epochs = epochs
        self.seed = seed
        self.train_crop_size = train_crop_size
        self.train_random_flip = bool(train_random_flip)
        self.train_random_rot90 = bool(train_random_rot90)
        self.grad_clip_norm = grad_clip_norm

        self.num_workers = resolve_num_workers(num_workers)
        self.output_dir = Path(output_dir)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.cache_root = resolve_cache_root(cache_dir)

        self.include_metadata = True
        self.pin_memory = True
        # Trainer rebuilds loaders per run, so keep worker lifetime on the PyTorch default path.
        self.persistent_workers = False
        self.prefetch_factor = 2
        self.drop_last = False
        self.current_epoch = 0
        self.global_step = 0
        self._config_written = False
        self._cache_ready: set[str] = set()
        self._prepared_split_states: dict[tuple[str, int | None], _PreparedSplitCacheEntry] = {}

        self.device = self._infer_module_device(self.model)
        self._wrap_model_for_ddp_if_needed()
        self.device = self._infer_module_device(self._model_state_owner())
        self.multiprocessing_context = self._resolve_multiprocessing_context(
            multiprocessing_context
        )

    def step(self) -> dict[str, Any]:
        """Run one training epoch and validation."""
        if self.current_epoch >= self.epochs:
            raise RuntimeError(
                f"all epochs are already consumed ({self.current_epoch}/{self.epochs})"
            )

        step_started_at = time.perf_counter()
        epoch_index = self.current_epoch
        self._seed_epoch(epoch_index)
        self._write_config_once()
        self._ensure_training_startup_caches()

        train_lrs = self._get_learning_rates()
        train_started_at = time.perf_counter()
        train_summary = self._run_training_epoch(epoch_index)
        train_elapsed_sec = time.perf_counter() - train_started_at
        train_lrs = train_summary.get("lr", train_lrs)
        train_summary["lr"] = train_lrs
        if is_primary():
            self._write_record(
                {
                    "kind": "train_epoch",
                    "epoch": epoch_index + 1,
                    **train_summary,
                }
            )
            tqdm.write(
                format_train_epoch_row(
                    epoch=epoch_index + 1,
                    epochs=self.epochs,
                    train=train_summary,
                    elapsed_sec=train_elapsed_sec,
                )
            )

        validation_started_at = time.perf_counter()
        validation_summary = self._run_evaluation(
            split="validation",
            max_samples=self.max_val_samples,
            epoch_index=epoch_index,
            description=f"val {epoch_index + 1}/{self.epochs}",
        )
        validation_elapsed_sec = time.perf_counter() - validation_started_at
        self._step_scheduler_after_validation(validation_summary)
        if is_primary():
            self._write_record(
                {
                    "kind": "validation",
                    "epoch": epoch_index + 1,
                    **validation_summary,
                }
            )
            tqdm.write(
                format_val_epoch_row(
                    epochs=self.epochs,
                    val=validation_summary,
                    train_learning_rates=train_lrs,
                    elapsed_sec=validation_elapsed_sec,
                )
            )

        elapsed_sec = time.perf_counter() - step_started_at

        self.current_epoch += 1
        result = {
            "epoch": epoch_index + 1,
            "train": train_summary,
            "val": validation_summary,
            "elapsed_sec": elapsed_sec,
        }
        return result

    def test(self) -> dict[str, Any]:
        """Run the test split with the current model state."""
        self._write_config_once()
        test_lrs = self._get_learning_rates()
        test_started_at = time.perf_counter()
        test_summary = self._run_evaluation(
            split="test",
            max_samples=self.max_test_samples,
            epoch_index=max(self.current_epoch - 1, 0),
            description="test",
        )
        test_elapsed_sec = time.perf_counter() - test_started_at
        if is_primary():
            self._write_record(
                {
                    "kind": "test",
                    "epoch": self.current_epoch,
                    **test_summary,
                }
            )

        result = {"epoch": self.current_epoch, **test_summary}
        if is_primary():
            tqdm.write(
                format_test_summary(
                    result,
                    learning_rates=test_lrs,
                    elapsed_sec=test_elapsed_sec,
                )
            )
        return result

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        """Persist model, optimizer, and runtime counters for resuming training."""
        self._write_config_once()
        checkpoint_path = self._resolve_artifact_path(
            path,
            default_name=f"epoch-{self.current_epoch:04d}.pt",
        )
        if not is_primary():
            return checkpoint_path

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model": self._model_state_owner().state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        torch.save(checkpoint, checkpoint_path)
        self._write_record(
            {
                "kind": "checkpoint_save",
                "path": checkpoint_path,
                "epoch": self.current_epoch,
                "global_step": self.global_step,
            }
        )
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore model, optimizer, and runtime counters from a checkpoint file."""
        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint must be a mapping produced by save_checkpoint()")
        if "model" not in checkpoint:
            raise KeyError("checkpoint is missing the 'model' state dict")
        if "optimizer" not in checkpoint:
            raise KeyError("checkpoint is missing the 'optimizer' state dict")

        self._model_state_owner().load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Legacy checkpoints predate scheduler support, so keep the caller's
        # scheduler state unchanged when no scheduler payload is available.
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.current_epoch = int(checkpoint.get("epoch", 0))
        self.global_step = int(checkpoint.get("global_step", 0))

        self._write_config_once()
        if is_primary():
            self._write_record(
                {
                    "kind": "checkpoint_load",
                    "path": checkpoint_path,
                    "epoch": self.current_epoch,
                    "global_step": self.global_step,
                }
            )
        return {
            "path": checkpoint_path,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }

    def save_weights(self, path: str | Path | None = None) -> Path:
        """Persist model weights without optimizer or runtime state."""
        weights_path = self._resolve_artifact_path(
            path,
            default_name=f"model-epoch-{self.current_epoch:04d}.pt",
        )
        if not is_primary():
            return weights_path

        torch.save(self._model_state_owner().state_dict(), weights_path)
        return weights_path

    def load_weights(self, path: str | Path, *, strict: bool = True) -> None:
        """Restore model weights from a weights file or checkpoint file."""
        loaded = torch.load(Path(path), map_location=self.device)
        state_dict: Any = loaded
        if isinstance(loaded, Mapping) and "model" in loaded and "optimizer" in loaded:
            state_dict = loaded["model"]
        self._model_state_owner().load_state_dict(state_dict, strict=strict)

    def predict(self, batch: Mapping[str, Any]) -> Any:
        """Run inference for a single batch without mutating training mode."""
        moved_batch = move_batch_to_device(dict(batch), self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self.model(moved_batch["sar"], moved_batch["cloudy"])
        finally:
            self.model.train(was_training)

    def get_state(self) -> dict[str, Any]:
        """Return the trainer runtime counters and execution context."""
        return {
            "epoch": self.current_epoch,
            "epochs": self.epochs,
            "global_step": self.global_step,
            "lr": self._get_learning_rates(),
            "device": self.device,
            "distributed": is_distributed(),
        }

    @staticmethod
    def _validate_max_samples(name: str, value: int | None) -> None:
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be greater than zero when provided")

    def _validate_optimizer_matches_model(self) -> None:
        model_param_ids = {id(parameter) for parameter in self.model.parameters()}
        optimizer_param_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if not optimizer_param_ids:
            raise ValueError("optimizer must contain model parameters")
        if not optimizer_param_ids.issubset(model_param_ids):
            raise ValueError("optimizer must be constructed from the provided model parameters")

    def _validate_scheduler_matches_optimizer(self) -> None:
        if self.scheduler is None:
            return
        if self.scheduler.optimizer is not self.optimizer:
            raise ValueError("scheduler must be constructed from the provided optimizer")

    def _resolve_scheduler_timing(self, value: SchedulerTiming) -> SchedulerTiming:
        normalized = value.strip().lower()
        if normalized not in _SCHEDULER_TIMING_CHOICES:
            supported = ", ".join(sorted(_SCHEDULER_TIMING_CHOICES))
            raise ValueError(f"scheduler_timing must be one of {supported}")
        return cast(SchedulerTiming, normalized)

    def _resolve_scheduler_config(
        self,
        *,
        scheduler_timing: SchedulerTiming,
        scheduler_monitor: str | None,
    ) -> _ResolvedSchedulerConfig:
        timing = self._resolve_scheduler_timing(scheduler_timing)
        if self.scheduler is None:
            if scheduler_monitor is not None:
                raise ValueError("scheduler_monitor requires a scheduler")
            if timing != "after_validation":
                raise ValueError(
                    "scheduler_timing requires a scheduler unless it is 'after_validation'"
                )
            return _ResolvedSchedulerConfig(timing=timing)

        if scheduler_monitor is not None and timing != "after_validation":
            raise ValueError(
                "scheduler_monitor is only supported when scheduler_timing is 'after_validation'"
            )

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if timing != "after_validation":
                raise ValueError(
                    "ReduceLROnPlateau requires scheduler_timing='after_validation'"
                )
            return _ResolvedSchedulerConfig(
                timing=timing,
                monitor=self._resolve_scheduler_monitor(scheduler_monitor),
            )

        if scheduler_monitor is not None:
            raise ValueError("scheduler_monitor is only supported for ReduceLROnPlateau")

        return _ResolvedSchedulerConfig(timing=timing)

    def _resolve_scheduler_monitor(
        self,
        value: str | None,
    ) -> _ResolvedSchedulerMonitor:
        path = "val.loss" if value is None else value.strip()
        if not path:
            raise ValueError("scheduler_monitor must not be empty")
        if path == "val.loss":
            return _ResolvedSchedulerMonitor(path=path)

        metrics_prefix = "val.metrics."
        if not path.startswith(metrics_prefix):
            raise ValueError(
                "scheduler_monitor must be 'val.loss' or 'val.metrics.<name>'"
            )

        metric_name = path[len(metrics_prefix):]
        if not metric_name:
            raise ValueError(
                "scheduler_monitor must be 'val.loss' or 'val.metrics.<name>'"
            )
        if metric_name not in self.metric_fns:
            raise ValueError(
                f"scheduler_monitor metric '{metric_name}' must match a configured metric"
            )
        return _ResolvedSchedulerMonitor(path=path, metric_name=metric_name)

    @staticmethod
    def _infer_module_device(module: nn.Module) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
        return torch.device("cpu")

    def _resolve_multiprocessing_context(self, value: str | None) -> str | None:
        normalized: str | None = None
        if value is not None:
            normalized = value.strip().lower()
            if normalized not in _MULTIPROCESSING_CONTEXT_CHOICES:
                supported = ", ".join(sorted(_MULTIPROCESSING_CONTEXT_CHOICES))
                raise ValueError(
                    f"multiprocessing_context must be one of {supported}, or None"
                )
        if self.num_workers <= 0:
            return None
        if normalized is not None:
            return normalized
        if self.device.type == "cuda":
            return "spawn"
        return None

    def _get_learning_rates(self) -> list[float]:
        return [
            float(lr)
            for param_group in self.optimizer.param_groups
            if (lr := param_group.get("lr")) is not None
        ]

    def _scheduler_name(self) -> str | None:
        if self.scheduler is None:
            return None
        return self.scheduler.__class__.__name__

    def _resolve_scheduler_monitor_value(
        self,
        validation_summary: Mapping[str, Any],
        monitor: _ResolvedSchedulerMonitor,
    ) -> float:
        if monitor.metric_name is None:
            return float(validation_summary["loss"])
        metrics = validation_summary["metrics"]
        return float(metrics[monitor.metric_name])

    def _step_scheduler_before_optimizer(self) -> None:
        if self.scheduler is None or self._scheduler_config.timing != "before_optimizer_step":
            return
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"Detected call of `lr_scheduler\.step\(\)` before "
                    r"`optimizer\.step\(\)`\..*"
                ),
                category=UserWarning,
            )
            self.scheduler.step()

    def _step_scheduler_after_optimizer(self) -> None:
        if self.scheduler is None or self._scheduler_config.timing != "after_optimizer_step":
            return
        self.scheduler.step()

    def _step_scheduler_after_validation(
        self,
        validation_summary: Mapping[str, Any],
    ) -> None:
        if self.scheduler is None or self._scheduler_config.timing != "after_validation":
            return
        monitor = self._scheduler_config.monitor
        if monitor is None:
            self.scheduler.step()
            return
        self.scheduler.step(self._resolve_scheduler_monitor_value(validation_summary, monitor))

    def _seed_epoch(self, epoch_index: int) -> None:
        seed_everything(self.seed + epoch_index)

    def _ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_artifact_path(self, path: str | Path | None, *, default_name: str) -> Path:
        artifact_path = Path(path) if path is not None else self.output_dir / default_name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        return artifact_path

    def _reset_metrics_file(self) -> None:
        if not is_primary():
            return
        self._ensure_output_dir()
        self.metrics_path.write_text("", encoding="utf-8")

    def _write_config_once(self) -> None:
        if self._config_written:
            return
        self._config_written = True

        if not is_primary():
            return

        self._reset_metrics_file()
        self._write_record(
            {
                "kind": "config",
                "dataset_name": DATASET_ID,
                "max_train_samples": self.max_train_samples,
                "max_val_samples": self.max_val_samples,
                "max_test_samples": self.max_test_samples,
                "seed": self.seed,
                "batch_size": self.batch_size,
                "accum_steps": self.accum_steps,
                "epochs": self.epochs,
                "num_workers": self.num_workers,
                "multiprocessing_context": self.multiprocessing_context,
                "scheduler": self._scheduler_name(),
                "scheduler_timing": self.scheduler_timing,
                "scheduler_monitor": self.scheduler_monitor,
                "train_crop_size": self.train_crop_size,
                "train_random_flip": self.train_random_flip,
                "train_random_rot90": self.train_random_rot90,
                "grad_clip_norm": self.grad_clip_norm,
            }
        )
        tqdm.write(
            format_config_banner(
                dataset_name=DATASET_ID,
                max_train_samples=self.max_train_samples,
                max_val_samples=self.max_val_samples,
                max_test_samples=self.max_test_samples,
                batch_size=self.batch_size,
                accum_steps=self.accum_steps,
                epochs=self.epochs,
                seed=self.seed,
                device=self.device,
                num_workers=self.num_workers,
                multiprocessing_context=self.multiprocessing_context,
                scheduler_name=self._scheduler_name(),
                scheduler_timing=self.scheduler_timing,
                scheduler_monitor=self.scheduler_monitor,
                grad_clip_norm=self.grad_clip_norm,
            )
        )

    def _wrap_model_for_ddp_if_needed(self) -> None:
        if not is_distributed() or isinstance(self.model, DDP):
            return
        if self.device.type == "cuda":
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self.model = DDP(self.model, device_ids=[device_index])
            return
        self.model = DDP(self.model)

    def _model_state_owner(self) -> nn.Module:
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def _ensure_split_cache(self, *, split: str, max_samples: int | None) -> None:
        if split in self._cache_ready:
            return

        ensure_split_cache(
            split=split,
            dataset_name=DATASET_ID,
            revision=None,
            max_samples=max_samples,
            seed=self.seed,
            cache_root=self.cache_root,
            startup_callback=self._handle_startup_event,
        )
        self._cache_ready.add(split)

    def _ensure_training_startup_caches(self) -> None:
        for split, max_samples in (
            ("train", self.max_train_samples),
            ("validation", self.max_val_samples),
            ("test", self.max_test_samples),
        ):
            self._ensure_split_cache(split=split, max_samples=max_samples)

    def _resolve_prepared_split_state(
        self,
        *,
        split: str,
        max_samples: int | None,
    ) -> PreparedSplitState:
        key = (split, max_samples)
        cached = self._prepared_split_states.get(key)
        if cached is not None:
            return cached.state

        self._ensure_split_cache(split=split, max_samples=max_samples)
        state = resolve_prepared_split_state(
            split=split,
            dataset_name=DATASET_ID,
            revision=None,
            max_samples=max_samples,
            seed=self.seed,
            cache_root=self.cache_root,
        )
        self._prepared_split_states[key] = _PreparedSplitCacheEntry(
            split=split,
            max_samples=max_samples,
            state=state,
        )
        return state

    def _build_loader(
        self,
        *,
        split: str,
        max_samples: int | None,
        training: bool,
        epoch_index: int,
    ) -> tuple[Any, int]:
        state = self._resolve_prepared_split_state(split=split, max_samples=max_samples)
        prepared = prepare_split_from_state(
            state,
            epoch=epoch_index,
            training=training,
            startup_callback=self._handle_startup_event,
        )
        loader = run_startup_stage(
            self._handle_startup_event,
            stage="build dataloader",
            split=split,
            operation=lambda: build_dataloader(
                prepared,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                training=training,
                seed=self.seed,
                epoch=epoch_index,
                include_metadata=self.include_metadata,
                pin_memory=self.pin_memory,
                multiprocessing_context=self.multiprocessing_context,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                drop_last=self.drop_last,
                crop_size=self.train_crop_size if training else None,
                crop_mode="random" if training and self.train_crop_size is not None else "none",
                random_flip=self.train_random_flip if training else False,
                random_rot90=self.train_random_rot90 if training else False,
            ),
            max_samples=max_samples,
        )
        return loader, self._infer_total_batches(
            num_examples=prepared.num_examples,
            batch_size=self.batch_size,
            training=training,
        )

    def _infer_total_batches(
        self,
        *,
        num_examples: int,
        batch_size: int,
        training: bool,
    ) -> int:
        if num_examples <= 0:
            return 0
        if training and self.drop_last:
            return num_examples // batch_size
        return (num_examples + batch_size - 1) // batch_size

    def _should_step_optimizer(self, *, batch_index: int, total_batches: int) -> bool:
        is_accum_boundary = (batch_index + 1) % self.accum_steps == 0
        is_last_batch = batch_index + 1 == total_batches
        return is_accum_boundary or is_last_batch

    def _accum_window_size(self, *, batch_index: int, total_batches: int) -> int:
        window_start = batch_index - (batch_index % self.accum_steps)
        return min(self.accum_steps, total_batches - window_start)

    def _gradient_sync_context(self, *, sync_gradients: bool):
        if sync_gradients or not isinstance(self.model, DDP):
            return nullcontext()
        return self.model.no_sync()

    @staticmethod
    def _format_batch_meta_preview(batch: Mapping[str, Any]) -> str | None:
        meta = batch.get("meta")
        if not isinstance(meta, Mapping):
            return None

        preview_parts: list[str] = []
        for field in ("scene", "patch"):
            values = meta.get(field)
            if isinstance(values, (list, tuple)):
                rendered = [str(value) for value in values[:3]]
                if len(values) > 3:
                    rendered.append("...")
                preview_parts.append(f"{field}=[{', '.join(rendered)}]")
                continue
            if values is not None:
                preview_parts.append(f"{field}={values}")

        if not preview_parts:
            return None
        return " ".join(preview_parts)

    def _training_failure_context(
        self,
        *,
        epoch_index: int,
        batch_index: int,
        batch: Mapping[str, Any],
    ) -> str:
        parts = [
            f"epoch={epoch_index + 1}",
            f"batch_index={batch_index}",
            f"global_step={self.global_step}",
        ]
        meta_preview = self._format_batch_meta_preview(batch)
        if meta_preview is not None:
            parts.append(meta_preview)
        return ", ".join(parts)

    def _assert_finite_loss(
        self,
        *,
        loss: torch.Tensor,
        epoch_index: int,
        batch_index: int,
        batch: Mapping[str, Any],
    ) -> None:
        if bool(torch.isfinite(loss.detach()).all()):
            return

        context = self._training_failure_context(
            epoch_index=epoch_index,
            batch_index=batch_index,
            batch=batch,
        )
        raise FloatingPointError(
            f"non-finite training loss before backward: loss={float(loss.detach().cpu().item())}, {context}"
        )

    def _assert_finite_gradients(
        self,
        *,
        epoch_index: int,
        batch_index: int,
        batch: Mapping[str, Any],
    ) -> None:
        bad_parameters: list[str] = []
        for name, parameter in self._model_state_owner().named_parameters():
            grad = parameter.grad
            if grad is None or bool(torch.isfinite(grad).all()):
                continue
            bad_parameters.append(name)
            if len(bad_parameters) >= 3:
                break

        if not bad_parameters:
            return

        context = self._training_failure_context(
            epoch_index=epoch_index,
            batch_index=batch_index,
            batch=batch,
        )
        raise FloatingPointError(
            "non-finite gradients before optimizer.step: "
            f"parameters={', '.join(bad_parameters)}, {context}"
        )

    def _apply_gradient_clipping(self) -> None:
        if self.grad_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(
            self._model_state_owner().parameters(),
            max_norm=self.grad_clip_norm,
            error_if_nonfinite=True,
        )

    def _run_optimizer_update(
        self,
        *,
        first_update_lrs: list[float] | None,
    ) -> list[float]:
        if self.scheduler_timing == "before_optimizer_step":
            self._step_scheduler_before_optimizer()

        applied_lrs = self._get_learning_rates() if first_update_lrs is None else first_update_lrs
        self.optimizer.step()

        if self.scheduler_timing == "after_optimizer_step":
            self._step_scheduler_after_optimizer()

        self.global_step += 1
        return applied_lrs

    def _run_training_epoch(self, epoch_index: int) -> dict[str, Any]:
        loader, total_batches = self._build_loader(
            split="train",
            max_samples=self.max_train_samples,
            training=True,
            epoch_index=epoch_index,
        )
        self._set_sampler_epoch(loader, epoch_index)
        self.model.train()
        self._handle_startup_event(
            {
                "stage": "start epoch",
                "split": "train",
                "status": "start",
                "epoch": epoch_index + 1,
            }
        )

        accumulator = MetricAccumulator()
        start_time = time.perf_counter()
        first_update_lrs: list[float] | None = None
        batch_iterator = self._prime_loader(split="train", loader=loader, max_samples=self.max_train_samples)
        progress = self._create_progress_bar(
            total=total_batches,
            description=f"train {epoch_index + 1}/{self.epochs}",
        )
        try:
            self.optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(batch_iterator):
                moved_batch = move_batch_to_device(batch, self.device)
                sync_gradients = self._should_step_optimizer(
                    batch_index=batch_index,
                    total_batches=total_batches,
                )
                accum_window_size = self._accum_window_size(
                    batch_index=batch_index,
                    total_batches=total_batches,
                )
                with self._gradient_sync_context(sync_gradients=sync_gradients):
                    model_output = self.model(moved_batch["sar"], moved_batch["cloudy"])
                    loss = compute_loss(self.loss_fn, model_output, moved_batch, self.device)
                    self._assert_finite_loss(
                        loss=loss,
                        epoch_index=epoch_index,
                        batch_index=batch_index,
                        batch=moved_batch,
                    )
                    (loss / accum_window_size).backward()

                if sync_gradients:
                    self._assert_finite_gradients(
                        epoch_index=epoch_index,
                        batch_index=batch_index,
                        batch=moved_batch,
                    )
                    self._apply_gradient_clipping()
                    first_update_lrs = self._run_optimizer_update(
                        first_update_lrs=first_update_lrs
                    )
                    self.optimizer.zero_grad(set_to_none=True)

                batch_size = int(moved_batch["sar"].shape[0])

                metric_values = compute_metric_values(self.metric_fns, model_output, moved_batch)
                batch_values = {"loss": loss.item(), **metric_values}
                accumulator.update(batch_values, batch_size)
                update_progress_bar(
                    progress,
                    accumulator=accumulator,
                    start_time=start_time,
                    reduce_int=self._reduce_int,
                    reduce_sum=self._reduce_sum,
                    distributed=is_distributed(),
                    learning_rates=self._get_learning_rates(),
                )
        finally:
            progress.close()

        summary = finalize_summary(
            accumulator=accumulator,
            start_time=start_time,
            include_speed=True,
            reduce_int=self._reduce_int,
            reduce_sum=self._reduce_sum,
            distributed=is_distributed(),
        )
        if summary["num_samples"] == 0:
            raise RuntimeError("training epoch produced no batches")
        if first_update_lrs is not None:
            summary["lr"] = first_update_lrs
        return summary

    def _run_evaluation(
        self,
        *,
        split: str,
        max_samples: int | None,
        epoch_index: int,
        description: str,
    ) -> dict[str, Any]:
        loader, total_batches = self._build_loader(
            split=split,
            max_samples=max_samples,
            training=False,
            epoch_index=epoch_index,
        )
        self.model.eval()

        accumulator = MetricAccumulator()
        start_time = time.perf_counter()
        batch_iterator = self._prime_loader(split=split, loader=loader, max_samples=max_samples)
        progress = self._create_progress_bar(total=total_batches, description=description)
        try:
            with torch.no_grad():
                for batch in batch_iterator:
                    moved_batch = move_batch_to_device(batch, self.device)
                    model_output = self.model(moved_batch["sar"], moved_batch["cloudy"])
                    loss = compute_loss(self.loss_fn, model_output, moved_batch, self.device)
                    batch_size = int(moved_batch["sar"].shape[0])

                    metric_values = compute_metric_values(self.metric_fns, model_output, moved_batch)
                    batch_values = {"loss": loss.item(), **metric_values}
                    accumulator.update(batch_values, batch_size)
                    update_progress_bar(
                        progress,
                        accumulator=accumulator,
                        start_time=start_time,
                        reduce_int=self._reduce_int,
                        reduce_sum=self._reduce_sum,
                        distributed=is_distributed(),
                    )
        finally:
            progress.close()

        summary = finalize_summary(
            accumulator=accumulator,
            start_time=None,
            include_speed=False,
            reduce_int=self._reduce_int,
            reduce_sum=self._reduce_sum,
            distributed=is_distributed(),
        )
        if summary["num_samples"] == 0:
            raise RuntimeError(f"{split} evaluation produced no batches")
        return summary

    def _create_progress_bar(self, *, total: int | None, description: str):
        is_training = description.startswith("train ")
        return tqdm(
            total=total,
            desc=description,
            disable=not is_primary(),
            ncols=resolve_progress_bar_ncols(),
            leave=False,
            colour="#4caf50" if is_training else "#00bcd4",
            smoothing=0.1,
            mininterval=0.1,
        )

    def _set_sampler_epoch(self, loader: Any, epoch_index: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch_index)

    def _reduce_sum(self, value: float) -> float:
        if not is_distributed():
            return value

        tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
        assert dist is not None
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item())

    def _reduce_int(self, value: int) -> int:
        if not is_distributed():
            return value

        tensor = torch.tensor(value, device=self.device, dtype=torch.int64)
        assert dist is not None
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def _write_record(self, record: Mapping[str, Any]) -> None:
        self._ensure_output_dir()
        serialized = {key: serialize_value(value) for key, value in record.items()}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serialized, sort_keys=True) + "\n")

    def _handle_startup_event(self, event: Mapping[str, Any]) -> None:
        if not is_primary():
            return

        record = {"kind": "startup", **event}
        self._write_record(record)
        if not should_print_startup(record):
            return
        tqdm.write(format_startup_message(record))

    def _prime_loader(
        self,
        *,
        split: str,
        loader: torch.utils.data.DataLoader,
        max_samples: int | None,
    ):
        primed = run_startup_stage(
            self._handle_startup_event,
            stage="wait first batch",
            split=split,
            operation=lambda: prime_iterator(loader),
            max_samples=max_samples,
        )
        if primed is None:
            return iter(())

        first_batch, remainder = primed
        return chain([first_batch], remainder)
