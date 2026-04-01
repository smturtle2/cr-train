from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm.auto import tqdm

from .data import (
    DATASET_ID,
    build_dataloader,
    ensure_split_cache,
    is_distributed,
    is_primary,
    move_batch_to_device,
    prepare_split,
    resolve_cache_root,
    resolve_num_workers,
    run_startup_stage,
    seed_everything,
)
from .trainer_reporting import format_config_banner, format_epoch_summary, format_startup_message, format_test_summary, serialize_value, should_print_startup
from .trainer_runtime import MetricAccumulator, compute_loss, compute_metric_values, finalize_summary, prime_iterator, update_progress_bar

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


LossFn = Callable[[Any, Mapping[str, Any]], torch.Tensor | float | int]
MetricFn = Callable[[Any, Mapping[str, Any]], torch.Tensor | float | int]


class Trainer:
    """SEN12MS-CR trainer.

    `dataset_seed` fixes the canonical shuffled dataset stream.
    `seed` drives the sample-selection planner over that canonical stream.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: LossFn,
        metrics: Mapping[str, MetricFn] | None = None,
        *,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        batch_size: int = 4,
        epochs: int = 1,
        seed: int = 42,
        dataset_seed: int | None = None,
        output_dir: str | Path = "runs/default",
        cache_dir: str | Path | None = None,
        predecoded: bool = False,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer instance")
        if not callable(loss):
            raise TypeError("loss must be callable")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero")

        self._validate_max_samples("max_train_samples", max_train_samples)
        self._validate_max_samples("max_val_samples", max_val_samples)
        self._validate_max_samples("max_test_samples", max_test_samples)

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss
        self.metric_fns = dict(metrics or {})
        for name, metric_fn in self.metric_fns.items():
            if not callable(metric_fn):
                raise TypeError(f"metric '{name}' must be callable")
        self._validate_optimizer_matches_model()

        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.dataset_seed = dataset_seed

        self.predecoded = predecoded
        self.num_workers = resolve_num_workers("auto")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.cache_root = resolve_cache_root(cache_dir)
        self._reset_metrics_file()

        self.include_metadata = True
        self.pin_memory = True
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = 2
        self.drop_last = False
        self.current_epoch = 0
        self.global_step = 0
        self._config_written = False
        self._cache_ready = False

        # DDP 래핑 후 원본 모듈에서 device 추론
        self._wrap_model_for_ddp_if_needed()
        self.device = self._infer_module_device(self._model_state_owner())

    def step(self) -> dict[str, Any]:
        """Run one training epoch, validation, and checkpoint the result."""
        if self.current_epoch >= self.epochs:
            raise RuntimeError(
                f"all epochs are already consumed ({self.current_epoch}/{self.epochs})"
            )

        epoch_index = self.current_epoch
        self._seed_epoch(epoch_index)
        self._write_config_once()
        self._ensure_split_caches()

        train_summary = self._run_training_epoch(epoch_index)
        if is_primary():
            self._write_record(
                {
                    "kind": "train_epoch",
                    "epoch": epoch_index + 1,
                    **train_summary,
                }
            )

        validation_summary = self._run_evaluation(
            split="validation",
            max_samples=self.max_val_samples,
            epoch_index=epoch_index,
            description=f"val {epoch_index + 1}/{self.epochs}",
        )
        if is_primary():
            self._write_record(
                {
                    "kind": "validation",
                    "epoch": epoch_index + 1,
                    **validation_summary,
                }
            )

        checkpoint_path = self._save_checkpoint(epoch_index + 1)
        if is_primary():
            self._write_record(
                {
                    "kind": "checkpoint",
                    "epoch": epoch_index + 1,
                    "path": checkpoint_path,
                }
            )

        self.current_epoch += 1
        result = {
            "epoch": epoch_index + 1,
            "train": train_summary,
            "val": validation_summary,
            "checkpoint_path": str(checkpoint_path),
        }
        if is_primary():
            tqdm.write(format_epoch_summary(result, epochs=self.epochs))
        return result

    def test(self) -> dict[str, Any]:
        """Run the test split with the current model state."""
        self._write_config_once()
        self._ensure_split_caches()
        test_summary = self._run_evaluation(
            split="test",
            max_samples=self.max_test_samples,
            epoch_index=max(self.current_epoch - 1, 0),
            description="test",
        )
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
            tqdm.write(format_test_summary(result))
        return result

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

    @staticmethod
    def _infer_module_device(module: nn.Module) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
        return torch.device("cpu")

    def _seed_epoch(self, epoch_index: int) -> None:
        seed_everything(self.seed + epoch_index, deterministic=True)

    def _reset_metrics_file(self) -> None:
        if not is_primary():
            return
        self.metrics_path.write_text("", encoding="utf-8")

    def _write_config_once(self) -> None:
        if self._config_written:
            return
        self._config_written = True

        if not is_primary():
            return

        self._write_record(
            {
                "kind": "config",
                "dataset_name": DATASET_ID,
                "max_train_samples": self.max_train_samples,
                "max_val_samples": self.max_val_samples,
                "max_test_samples": self.max_test_samples,
                "seed": self.seed,
                "dataset_seed": self.dataset_seed,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "num_workers": self.num_workers,
            }
        )
        tqdm.write(format_config_banner(
            dataset_name=DATASET_ID,
            max_train_samples=self.max_train_samples,
            max_val_samples=self.max_val_samples,
            max_test_samples=self.max_test_samples,
            batch_size=self.batch_size,
            epochs=self.epochs,
            seed=self.seed,
            dataset_seed=self.dataset_seed,
            device=self.device,
        ))

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

    def _ensure_split_caches(self) -> None:
        if self._cache_ready:
            return

        split_specs = (
            ("train", self.max_train_samples),
            ("validation", self.max_val_samples),
            ("test", self.max_test_samples),
        )
        for split, max_samples in split_specs:
            ensure_split_cache(
                split=split,
                dataset_name=DATASET_ID,
                revision=None,
                max_samples=max_samples,
                seed=self.seed,
                dataset_seed=self.dataset_seed,
                cache_root=self.cache_root,
                startup_callback=self._handle_startup_event,
                predecoded=self.predecoded,
            )
        self._cache_ready = True

    def _build_loader(
        self,
        *,
        split: str,
        max_samples: int | None,
        training: bool,
        epoch_index: int,
    ) -> tuple[Any, int | None]:
        prepared = prepare_split(
            split=split,
            dataset_name=DATASET_ID,
            revision=None,
            max_samples=max_samples,
            seed=self.seed,
            dataset_seed=self.dataset_seed,
            cache_root=self.cache_root,
            startup_callback=self._handle_startup_event,
            predecoded=self.predecoded,
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
                predecoded=self.predecoded,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                drop_last=self.drop_last,
            ),
            max_samples=max_samples,
        )
        return loader, self._infer_total_batches(loader)

    def _infer_total_batches(self, loader: Any) -> int | None:
        return len(loader)

    def _run_training_epoch(self, epoch_index: int) -> dict[str, Any]:
        loader, total_batches = self._build_loader(
            split="train",
            max_samples=self.max_train_samples,
            training=True,
            epoch_index=epoch_index,
        )
        self._set_sampler_epoch(loader, epoch_index)
        self.model.train()
        batch_iterator = self._prime_loader(split="train", loader=loader, max_samples=self.max_train_samples)
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
        progress = self._create_progress_bar(
            total=total_batches,
            description=f"train {epoch_index + 1}/{self.epochs}",
        )
        try:
            for batch in batch_iterator:
                moved_batch = move_batch_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                model_output = self.model(moved_batch["sar"], moved_batch["cloudy"])
                loss = compute_loss(self.loss_fn, model_output, moved_batch, self.device)
                loss.backward()
                self.optimizer.step()

                batch_size = int(moved_batch["sar"].shape[0])
                self.global_step += 1

                metric_values = compute_metric_values(self.metric_fns, model_output, moved_batch)
                batch_values = {"loss": loss.item(), **metric_values}
                accumulator.update(batch_values, batch_size)
                update_progress_bar(
                    progress,
                    accumulator=accumulator,
                    start_time=start_time,
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
        batch_iterator = self._prime_loader(split=split, loader=loader, max_samples=max_samples)

        accumulator = MetricAccumulator()
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
                        start_time=None,
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
            dynamic_ncols=True,
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

    def _save_checkpoint(self, next_epoch: int) -> Path:
        checkpoint_path = self.output_dir / f"epoch-{next_epoch:04d}.pt"
        if not is_primary():
            return checkpoint_path

        checkpoint = {
            "epoch": next_epoch,
            "global_step": self.global_step,
            "model": self._model_state_owner().state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def _write_record(self, record: Mapping[str, Any]) -> None:
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
