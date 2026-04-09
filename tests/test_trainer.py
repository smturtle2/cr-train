from __future__ import annotations

import copy
import importlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cr_train import Trainer
from cr_train.data import BLOCK_SIZE
from cr_train.progress import resolve_progress_bar_ncols
from cr_train.data.store import resolve_block_cache_paths
from cr_train.trainer_reporting import (
    format_cache_summary,
    format_epoch_summary,
    format_startup_message,
    format_test_summary,
    format_train_epoch_row,
    format_val_epoch_row,
)


def _make_row(index: int) -> dict[str, object]:
    sar = torch.arange(256 * 256 * 2, dtype=torch.float32).reshape(256, 256, 2).numpy() + index
    cloudy = torch.arange(256 * 256 * 13, dtype=torch.int16).reshape(256, 256, 13).numpy() + index
    target = torch.arange(256 * 256 * 13, dtype=torch.int16).reshape(256, 256, 13).numpy() + index + 1
    return {
        "sar": sar.tobytes(),
        "cloudy": cloudy.tobytes(),
        "target": target.tobytes(),
        "sar_shape": [256, 256, 2],
        "opt_shape": [256, 256, 13],
        "season": "summer",
        "scene": str(index),
        "patch": f"p{index:03d}",
    }


def _make_block_splits(block_count: int) -> list[list[dict[str, object]]]:
    blocks: list[list[dict[str, object]]] = []
    current_index = 0
    for _ in range(block_count):
        block_rows = [_make_row(current_index + offset) for offset in range(BLOCK_SIZE)]
        blocks.append(block_rows)
        current_index += BLOCK_SIZE
    return blocks


def _catalog(split: str, blocks: list[list[dict[str, object]]]) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    import hashlib

    rows_by_key: dict[str, list[dict[str, object]]] = {}
    block_entries = []
    total_rows = 0
    for index, rows in enumerate(blocks):
        cache_key = hashlib.sha256(f"{split}:{index}".encode("utf-8")).hexdigest()[:16]
        rows_by_key[cache_key] = [dict(row) for row in rows]
        block_entries.append(
            {
                "index": index,
                "shard_index": index,
                "cache_key": cache_key,
                "source_file": f"hf://datasets/unit/test/{split}/{index:04d}.parquet",
                "row_groups": [index],
            }
        )
        total_rows += len(rows)
    return {
        "split": split,
        "total_rows": total_rows,
        "total_blocks": len(block_entries),
        "blocks": block_entries,
    }, rows_by_key


def _patch_split_cache(monkeypatch, tmp_path: Path, split_blocks: dict[str, list[list[dict[str, object]]]]) -> dict[str, object]:
    source_root = tmp_path / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    catalogs: dict[str, dict[str, object]] = {}
    rows_by_key: dict[str, list[dict[str, object]]] = {}
    split_sizes: dict[str, int] = {}
    for split, blocks in split_blocks.items():
        catalog, block_rows = _catalog(split, blocks)
        catalogs[split] = catalog
        rows_by_key.update(block_rows)
        split_sizes[split] = int(catalog["total_rows"])

    descriptor = {
        "dataset_name": "unit/test",
        "revision": None,
        "split_sizes": split_sizes,
    }
    load_counts: dict[str, int] = defaultdict(int)

    def fake_ensure_source_root(*, dataset_name: str, revision: str | None, cache_root: Path):
        assert dataset_name in {"unit/test", "Hermanni/sen12mscr"}
        del revision, cache_root
        return source_root, descriptor

    def fake_ensure_split_catalog(*, source_root: Path, descriptor: dict[str, object], split: str, startup_callback):
        del source_root, descriptor, startup_callback
        return catalogs[split]

    def fake_load_block_rows(
        *,
        dataset_name: str,
        revision: str | None,
        split: str,
        block: dict[str, object],
        progress_callback=None,
    ):
        del dataset_name, revision, split
        cache_key = str(block["cache_key"])
        load_counts[cache_key] += 1
        rows = [dict(row) for row in rows_by_key[cache_key]]
        if progress_callback is not None:
            for index, row in enumerate(rows, start=1):
                downloaded_bytes = sum(
                    len(value)
                    for value in row.values()
                    if isinstance(value, (bytes, bytearray, memoryview))
                )
                progress_callback(index, downloaded_bytes)
        return rows

    monkeypatch.setattr("cr_train.data.runtime.ensure_source_root", fake_ensure_source_root)
    monkeypatch.setattr("cr_train.data.runtime.ensure_split_catalog", fake_ensure_split_catalog)
    monkeypatch.setattr("cr_train.data.runtime.load_block_rows", fake_load_block_rows)
    monkeypatch.setattr("cr_train.data.dataset.ensure_source_root", fake_ensure_source_root)
    monkeypatch.setattr("cr_train.data.dataset.ensure_split_catalog", fake_ensure_split_catalog)
    return {
        "source_root": source_root,
        "catalogs": catalogs,
        "load_counts": load_counts,
    }


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Conv2d(15, 13, kernel_size=1)

    def forward(self, sar, cloudy):
        return self.body(torch.cat([sar, cloudy], dim=1))


class FakeDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, device_ids=None) -> None:
        super().__init__()
        self.module = module
        self.device_ids = device_ids

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class CountingStepLR(torch.optim.lr_scheduler.StepLR):
    def __init__(self, optimizer: torch.optim.Optimizer, *, step_size: int, gamma: float) -> None:
        self.step_calls = 0
        super().__init__(optimizer, step_size=step_size, gamma=gamma)
        self.step_calls = 0

    def step(self, epoch=None) -> None:
        self.step_calls += 1
        if epoch is None:
            super().step()
            return
        super().step(epoch)


class CountingReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer: torch.optim.Optimizer, **kwargs) -> None:
        self.step_calls = 0
        self.monitor_values: list[float] = []
        super().__init__(optimizer, **kwargs)
        self.step_calls = 0
        self.monitor_values.clear()

    def step(self, metrics, epoch=None) -> None:
        self.step_calls += 1
        self.monitor_values.append(float(metrics))
        if epoch is None:
            super().step(metrics)
            return
        super().step(metrics, epoch)


def _clone_model_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right)
        return
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert set(left.keys()) == set(right.keys())
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def loss_fn(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.nn.functional.mse_loss(prediction, batch["target"])


def mae_metric(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - batch["target"]))


def _make_train_summary(
    *,
    loss: float = 0.1,
    metrics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    return {
        "loss": loss,
        "metrics": dict(metrics or {}),
        "num_samples": 8,
        "num_batches": 1,
        "samples_per_sec": 1.0,
        "batches_per_sec": 1.0,
    }


def _make_eval_summary(
    *,
    loss: float,
    metrics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    return {
        "loss": loss,
        "metrics": dict(metrics or {}),
        "num_samples": 4,
        "num_batches": 1,
    }


def _patch_epoch_execution(
    monkeypatch,
    *,
    train_summaries: list[dict[str, object]],
    validation_summaries: list[dict[str, object]],
) -> None:
    train_queue = iter(copy.deepcopy(train_summaries))
    validation_queue = iter(copy.deepcopy(validation_summaries))

    monkeypatch.setattr(Trainer, "_ensure_training_startup_caches", lambda self: None)

    def fake_run_training_epoch(self, epoch_index: int) -> dict[str, object]:
        del self, epoch_index
        return copy.deepcopy(next(train_queue))

    def fake_run_evaluation(
        self,
        *,
        split: str,
        max_samples: int | None,
        epoch_index: int,
        description: str,
    ) -> dict[str, object]:
        del self, max_samples, epoch_index, description
        assert split == "validation"
        return copy.deepcopy(next(validation_queue))

    monkeypatch.setattr(Trainer, "_run_training_epoch", fake_run_training_epoch)
    monkeypatch.setattr(Trainer, "_run_evaluation", fake_run_evaluation)


class FakeTqdm:
    instances: list["FakeTqdm"] = []
    writes: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.disable = kwargs.get("disable", False)
        self.ncols = kwargs.get("ncols")
        self.updates: list[int] = []
        self.postfixes: list[str] = []
        self.desc_history: list[str] = [str(self.desc)] if self.desc is not None else []
        FakeTqdm.instances.append(self)

    @staticmethod
    def write(message: str) -> None:
        FakeTqdm.writes.append(message)

    def update(self, value: int) -> None:
        self.updates.append(value)

    def set_postfix(self, values) -> None:
        self.postfixes.append(str(values))

    def set_postfix_str(self, text: str) -> None:
        self.postfixes.append(text)

    def set_description_str(self, desc: str, refresh: bool = True) -> None:
        del refresh
        self.desc = desc
        self.desc_history.append(desc)

    def close(self) -> None:
        return None


def _elapsed_column_start(text: str) -> int:
    match = re.search(r"\d+\.\d+s$", text)
    assert match is not None
    return match.start()


def test_top_level_package_exports_trainer_as_primary_entry_point() -> None:
    package = importlib.import_module("cr_train")
    namespace: dict[str, object] = {}
    exec("from cr_train import Trainer\n", namespace)

    assert package.__all__ == ["Trainer"]
    assert package.Trainer is Trainer
    assert namespace["Trainer"] is Trainer


def test_format_cache_summary_compacts_square_timeline_on_one_line() -> None:
    summary = format_cache_summary(
        {
            "split": "train",
            "selected_block_count": 20,
            "selected_missing_blocks": 3,
            "resolved_blocks": 3,
            "elapsed_sec": 1.25,
            "timeline": ("█░" * 20),
        }
    )
    plain_summary = re.sub(r"\x1b\[[0-9;]*m", "", summary)

    assert "\n" not in summary
    assert "1.2s" in plain_summary
    assert "■□" in plain_summary
    assert "█" not in summary and "░" not in summary
    assert "…" in summary
    assert "cache train │ warm │ selected: 20, fill: 3/3 │ 1.2s" in plain_summary


def test_format_startup_message_uses_dim_separator() -> None:
    summary = format_startup_message(
        {
            "split": "train",
            "stage": "build dataloader",
            "status": "error",
            "max_samples": 128,
            "error": "boom",
        }
    )
    plain_summary = re.sub(r"\x1b\[[0-9;]*m", "", summary)

    assert plain_summary == "startup │ split=train │ stage=build dataloader │ max_samples=128 │ error=boom"


def test_resolve_progress_bar_ncols_leaves_one_column_headroom(monkeypatch) -> None:
    import cr_train.progress as progress_mod

    monkeypatch.setattr(
        progress_mod.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((80, 24)),
    )

    assert resolve_progress_bar_ncols(file=SimpleNamespace(isatty=lambda: True)) == 79


def test_resolve_progress_bar_ncols_returns_none_for_non_tty() -> None:
    assert resolve_progress_bar_ncols(file=SimpleNamespace(isatty=lambda: False)) is None


def test_format_epoch_summary_prefers_elapsed_time_over_throughput() -> None:
    summary = format_epoch_summary(
        {
            "epoch": 1,
            "train": {
                "loss": 0.0423,
                "metrics": {"mae": 0.0312},
                "lr": [1.234567e-5],
                "samples_per_sec": 142.3,
            },
            "val": {
                "loss": 0.0391,
                "metrics": {"mae": 0.0298},
            },
            "train_elapsed_sec": 12.34,
            "val_elapsed_sec": 2.43,
            "elapsed_sec": 14.77,
        },
        epochs=2,
    )
    plain_summary = re.sub(r"\x1b\[[0-9;]*m", "", summary)
    lines = plain_summary.splitlines()

    assert summary.count("\n") == 1
    assert "\t" not in summary
    assert len(re.findall(r"Epoch\s+1/\s+2", plain_summary)) == 1
    assert re.match(r"^Epoch\s+1/\s+2 │ train", lines[0])
    assert re.match(r"^\s+│ val\s+│ loss ", lines[1])
    assert re.search(r"loss\s+0\.0423", plain_summary)
    assert re.search(r"mae\s+0\.0312", plain_summary)
    assert re.search(r"val\s+│\s+loss\s+0\.0391\s+│\s+mae\s+0\.0298", plain_summary)
    assert lines[0].endswith("12.3s")
    assert lines[1].endswith("2.4s")
    assert re.search(r"lr\s+1\.2346e-5", plain_summary)
    assert "lr " not in lines[1]
    assert _elapsed_column_start(lines[0]) == _elapsed_column_start(lines[1])
    assert "samples/s" not in plain_summary
    assert " │ " in lines[0]
    assert " │ " in lines[1]


def test_format_epoch_rows_align_validation_elapsed_without_learning_rate() -> None:
    train_line = format_train_epoch_row(
        epoch=1,
        epochs=2,
        train={
            "loss": 0.0423,
            "metrics": {"mae": 0.0312},
            "lr": [1e-3],
        },
        elapsed_sec=12.34,
    )
    val_line = format_val_epoch_row(
        epochs=2,
        val={
            "loss": 0.0391,
            "metrics": {"mae": 0.0298},
        },
        train_learning_rates=[1e-3],
        elapsed_sec=2.43,
    )
    plain_train_line = re.sub(r"\x1b\[[0-9;]*m", "", train_line)
    plain_val_line = re.sub(r"\x1b\[[0-9;]*m", "", val_line)

    assert re.match(r"^Epoch\s+1/\s+2 │ train", plain_train_line)
    assert re.match(r"^\s+│ val\s+│ loss ", plain_val_line)
    assert re.search(r"lr\s+1e-3", plain_train_line)
    assert "lr " not in plain_val_line
    assert plain_train_line.endswith("12.3s")
    assert plain_val_line.endswith("2.4s")
    assert _elapsed_column_start(plain_train_line) == _elapsed_column_start(plain_val_line)
    assert plain_train_line.count(" │ ") == plain_val_line.count(" │ ")


def test_format_epoch_rows_keep_elapsed_aligned_across_learning_rate_width_changes() -> None:
    shorter_lr_line = format_train_epoch_row(
        epoch=4,
        epochs=10,
        train={
            "loss": 0.1074,
            "metrics": {"mae": 0.1074},
            "lr": [8.682e-4],
        },
        elapsed_sec=4.3,
    )
    longer_lr_line = format_train_epoch_row(
        epoch=3,
        epochs=10,
        train={
            "loss": 0.1407,
            "metrics": {"mae": 0.1407},
            "lr": [9.6575e-4],
        },
        elapsed_sec=4.3,
    )
    plain_shorter_lr_line = re.sub(r"\x1b\[[0-9;]*m", "", shorter_lr_line)
    plain_longer_lr_line = re.sub(r"\x1b\[[0-9;]*m", "", longer_lr_line)

    assert re.search(r"lr\s+8\.682e-4", plain_shorter_lr_line)
    assert re.search(r"lr\s+9\.6575e-4", plain_longer_lr_line)
    assert _elapsed_column_start(plain_shorter_lr_line) == _elapsed_column_start(plain_longer_lr_line)
    assert plain_shorter_lr_line.count(" │ ") == plain_longer_lr_line.count(" │ ")


def test_format_test_summary_uses_fixed_decimal_metrics() -> None:
    summary = format_test_summary(
        {
            "loss": 0.03912,
            "metrics": {"mae": 0.0},
            "num_samples": 256,
        },
        learning_rates=[1e-3],
        elapsed_sec=2.14,
    )
    plain_summary = re.sub(r"\x1b\[[0-9;]*m", "", summary)

    assert re.match(r"^\s+│ Test\s+│ loss ", plain_summary)
    assert re.search(r"loss\s+0\.0391", plain_summary)
    assert re.search(r"mae\s+0\.0000", plain_summary)
    assert "lr " not in plain_summary
    assert plain_summary.endswith("│ 2.1s")


def test_trainer_step_and_test_with_block_cache_warmup(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("old-record\n", encoding="utf-8")

    FakeTqdm.instances.clear()
    FakeTqdm.writes.clear()
    patched = _patch_split_cache(
        monkeypatch,
        tmp_path,
        {
            "train": _make_block_splits(4),
            "validation": _make_block_splits(4),
            "test": _make_block_splits(4),
        },
    )
    monkeypatch.setattr("cr_train.data.runtime.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.data.runtime.resolve_progress_bar_ncols", lambda: 79)
    monkeypatch.setattr("cr_train.trainer.resolve_progress_bar_ncols", lambda: 79)
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        metrics={"mae": mae_metric},
        max_train_samples=2 * BLOCK_SIZE,
        max_val_samples=BLOCK_SIZE,
        max_test_samples=BLOCK_SIZE,
        epochs=1,
        batch_size=8,
        seed=7,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    assert metrics_path.read_text(encoding="utf-8") == "old-record\n"

    epoch_summary = trainer.step()
    metrics_records_after_step = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    startup_records_after_step = [record for record in metrics_records_after_step if record["kind"] == "startup"]
    warmup_splits_after_step = {
        record["split"]
        for record in startup_records_after_step
        if record["stage"] == "warm split cache" and record["status"] == "done"
    }

    test_summary = trainer.test()
    metrics_records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    startup_records = [record for record in metrics_records if record["kind"] == "startup"]
    startup_stages = [record["stage"] for record in startup_records]
    batch_bars = [
        instance
        for instance in FakeTqdm.instances
        if str(instance.desc).startswith(("train", "val", "test"))
    ]
    train_bars = [instance for instance in batch_bars if str(instance.desc).startswith("train")]
    eval_bars = [instance for instance in batch_bars if str(instance.desc).startswith(("val", "test"))]
    warmup_bars = [
        instance
        for instance in FakeTqdm.instances
        if any("cache " in desc for desc in instance.desc_history)
    ]
    config_record = next(record for record in metrics_records if record["kind"] == "config")
    train_record = next(record for record in metrics_records if record["kind"] == "train_epoch")

    assert epoch_summary["epoch"] == 1
    assert epoch_summary["elapsed_sec"] > 0.0
    assert epoch_summary["train"]["loss"] >= 0.0
    assert "mae" in epoch_summary["train"]["metrics"]
    assert epoch_summary["train"]["lr"] == [1e-3]
    assert "checkpoint_path" not in epoch_summary
    assert not (output_dir / "epoch-0001.pt").exists()

    assert test_summary["epoch"] == 1
    assert "mae" in test_summary["metrics"]
    assert epoch_summary["train"]["num_batches"] == (2 * BLOCK_SIZE) // 8
    assert "batches_per_sec" in epoch_summary["train"]
    assert all(record["kind"] != "checkpoint" for record in metrics_records)
    assert "old-record" not in metrics_path.read_text(encoding="utf-8")
    assert train_record["lr"] == [1e-3]

    assert "dataset_seed" not in config_record
    assert config_record["multiprocessing_context"] is None
    assert config_record["scheduler"] is None
    assert config_record["scheduler_monitor"] is None
    assert config_record["train_crop_size"] == 128
    assert config_record["train_random_flip"] is True
    assert config_record["train_random_rot90"] is True
    assert warmup_splits_after_step == {"train", "validation", "test"}
    assert "warm split cache" in startup_stages
    assert "load local cache" in startup_stages
    assert "build dataloader" in startup_stages
    assert "wait first batch" in startup_stages
    assert "start epoch" in startup_stages

    assert len(FakeTqdm.writes) >= 7
    plain_writes = [re.sub(r"\x1b\[[0-9;]*m", "", message) for message in FakeTqdm.writes]
    assert any("cr-train" in message for message in FakeTqdm.writes)
    assert any(re.search(r"Epoch\s+1/\s+1", message) for message in plain_writes)
    assert any("Test" in message for message in FakeTqdm.writes)
    train_message_index = next(i for i, message in enumerate(plain_writes) if re.match(r"^Epoch\s+1/\s+1 │ train", message))
    val_message_index = next(i for i, message in enumerate(plain_writes) if re.match(r"^\s+│ val\s+│ loss ", message))
    test_message_index = next(i for i, message in enumerate(plain_writes) if re.match(r"^\s+│ Test\s+│ loss ", message))
    train_message = FakeTqdm.writes[train_message_index]
    val_message = FakeTqdm.writes[val_message_index]
    test_message = FakeTqdm.writes[test_message_index]
    plain_train_message = plain_writes[train_message_index]
    plain_val_message = plain_writes[val_message_index]
    plain_test_message = plain_writes[test_message_index]

    assert train_message_index < val_message_index
    assert train_message.count("\n") == 0
    assert val_message.count("\n") == 0
    assert "\t" not in train_message
    assert "\t" not in val_message
    assert len(re.findall(r"Epoch\s+1/\s+1", plain_train_message)) == 1
    assert re.search(r"lr\s+1e-3", plain_train_message)
    assert "lr " not in plain_val_message
    assert "lr " not in plain_test_message
    assert "samples/s" not in plain_train_message
    assert " │ " in plain_train_message
    assert " │ " in plain_val_message
    assert " │ " in plain_test_message
    assert _elapsed_column_start(plain_train_message) == _elapsed_column_start(plain_val_message)
    assert _elapsed_column_start(plain_val_message) == _elapsed_column_start(plain_test_message)
    assert test_message.count("\n") == 0
    warmup_messages = [message for message in FakeTqdm.writes if message.startswith("cache ")]
    assert len(warmup_messages) == 3
    assert all("\n" not in message for message in warmup_messages)
    assert all(("■" in message or "□" in message) for message in warmup_messages)
    assert all("█" not in message and "░" not in message for message in FakeTqdm.writes)

    assert len(warmup_bars) == 3
    assert all(int(bar.total) >= 1 for bar in warmup_bars)
    assert all(sum(bar.updates) == int(bar.total) for bar in warmup_bars)
    assert all(any("sel" in values for values in bar.postfixes) for bar in warmup_bars)
    assert all(all("blk/s" not in values for values in bar.postfixes) for bar in warmup_bars)
    assert all(any("MB/s" in values for values in bar.postfixes) for bar in warmup_bars)
    assert all(len(bar.postfixes) > int(bar.total) for bar in warmup_bars)
    assert all(len(bar.desc_history) == 1 for bar in warmup_bars)
    assert all(bar.ncols == 79 for bar in warmup_bars)

    assert len(batch_bars) == 3
    assert len(train_bars) == 1
    assert len(eval_bars) == 2
    assert any(
        any(
            re.search(r"loss: -?\d+\.\d{4}", values)
            and re.search(r"mae: -?\d+\.\d{4}", values)
            for values in bar.postfixes
        )
        for bar in batch_bars
    )
    assert all(any("lr: 1e-3" in values for values in bar.postfixes) for bar in train_bars)
    assert all(all("lr:" not in values for values in bar.postfixes) for bar in eval_bars)
    assert all(all("batches/s" not in values for values in bar.postfixes) for bar in batch_bars)
    assert all(all(value == 1 for value in bar.updates) for bar in batch_bars)
    assert all(bar.ncols == 79 for bar in batch_bars)

    warmup_done_records = [
        record
        for record in startup_records
        if record["stage"] == "warm split cache" and record["status"] == "done"
    ]
    assert [record["selected_block_count"] for record in warmup_done_records[:3]] == [2, 1, 1]
    assert all(record["planner_mode"] == "uniform_exact_k" for record in warmup_done_records[:3])
    assert all("timeline" in record for record in warmup_done_records[:3])
    assert all(len(str(record["timeline"])) == int(record["execution_block_count"]) for record in warmup_done_records[:3])
    assert all(set(str(record["timeline"])) <= {"█", "░"} for record in warmup_done_records[:3])
    assert all(record["resolved_blocks"] == record["selected_missing_blocks"] for record in warmup_done_records[:3])

    for split in ("train", "validation", "test"):
        cache_paths = resolve_block_cache_paths(patched["source_root"], split)
        metadata_records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in cache_paths.metadata_root.glob("*.json")
        ]
        assert metadata_records
        assert all("shard_index" in record for record in metadata_records)

    assert len(warmup_done_records) == 3
    assert sum(patched["load_counts"].values()) == 4

    with pytest.raises(RuntimeError):
        trainer.step()


def test_trainer_steps_scheduler_once_per_epoch_and_reports_learning_rate(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    _patch_split_cache(
        monkeypatch,
        tmp_path,
        {
            "train": _make_block_splits(4),
            "validation": _make_block_splits(4),
            "test": _make_block_splits(4),
        },
    )
    FakeTqdm.instances.clear()
    FakeTqdm.writes.clear()
    monkeypatch.setattr("cr_train.trainer.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.data.runtime.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CountingStepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        scheduler=scheduler,
        max_train_samples=2 * BLOCK_SIZE,
        max_val_samples=BLOCK_SIZE,
        max_test_samples=BLOCK_SIZE,
        epochs=2,
        batch_size=8,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    epoch_summary = trainer.step()
    metrics_records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    config_record = next(record for record in metrics_records if record["kind"] == "config")
    train_record = next(record for record in metrics_records if record["kind"] == "train_epoch")

    assert scheduler.step_calls == 1
    assert epoch_summary["train"]["lr"] == [1e-3]
    assert trainer.get_state()["lr"] == [5e-4]
    assert train_record["lr"] == [1e-3]
    assert config_record["scheduler"] == "CountingStepLR"
    assert config_record["scheduler_monitor"] is None
    assert any("scheduler CountingStepLR" in message for message in FakeTqdm.writes)


def test_trainer_steps_reduce_lr_on_plateau_with_default_val_loss_monitor(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    FakeTqdm.instances.clear()
    FakeTqdm.writes.clear()
    monkeypatch.setattr("cr_train.trainer.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)
    _patch_epoch_execution(
        monkeypatch,
        train_summaries=[_make_train_summary(), _make_train_summary()],
        validation_summaries=[
            _make_eval_summary(loss=0.5, metrics={"mae": 0.2}),
            _make_eval_summary(loss=0.6, metrics={"mae": 0.1}),
        ],
    )

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CountingReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
    )
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        metrics={"mae": mae_metric},
        scheduler=scheduler,
        epochs=2,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    first_epoch = trainer.step()
    second_epoch = trainer.step()
    metrics_records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    config_record = next(record for record in metrics_records if record["kind"] == "config")

    assert first_epoch["train"]["lr"] == [1e-3]
    assert second_epoch["train"]["lr"] == [1e-3]
    assert scheduler.step_calls == 2
    assert scheduler.monitor_values == [0.5, 0.6]
    assert trainer.get_state()["lr"] == [5e-4]
    assert config_record["scheduler"] == "CountingReduceLROnPlateau"
    assert config_record["scheduler_monitor"] == "val.loss"
    assert any("monitor val.loss" in message for message in FakeTqdm.writes)


def test_trainer_steps_reduce_lr_on_plateau_with_metric_monitor(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    FakeTqdm.writes.clear()
    monkeypatch.setattr("cr_train.trainer.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)
    _patch_epoch_execution(
        monkeypatch,
        train_summaries=[_make_train_summary(), _make_train_summary()],
        validation_summaries=[
            _make_eval_summary(loss=0.4, metrics={"mae": 0.2}),
            _make_eval_summary(loss=0.3, metrics={"mae": 0.3}),
        ],
    )

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CountingReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
    )
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        metrics={"mae": mae_metric},
        scheduler=scheduler,
        scheduler_monitor="val.metrics.mae",
        epochs=2,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    trainer.step()
    trainer.step()
    metrics_records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    config_record = next(record for record in metrics_records if record["kind"] == "config")

    assert scheduler.monitor_values == [0.2, 0.3]
    assert trainer.get_state()["lr"] == [5e-4]
    assert config_record["scheduler_monitor"] == "val.metrics.mae"
    assert any("monitor val.metrics.mae" in message for message in FakeTqdm.writes)


def test_trainer_save_and_load_checkpoint_round_trip_with_reduce_lr_on_plateau(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    FakeTqdm.writes.clear()
    monkeypatch.setattr("cr_train.trainer.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)
    _patch_epoch_execution(
        monkeypatch,
        train_summaries=[_make_train_summary(), _make_train_summary()],
        validation_summaries=[
            _make_eval_summary(loss=0.5),
            _make_eval_summary(loss=0.6),
        ],
    )

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CountingReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
    )
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        scheduler=scheduler,
        epochs=2,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    trainer.step()
    trainer.step()
    original_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    original_scheduler_state = copy.deepcopy(trainer.scheduler.state_dict())
    checkpoint_path = trainer.save_checkpoint()

    trainer.scheduler.step(2.0)
    trainer.optimizer.param_groups[0]["lr"] = 0.25
    trainer.current_epoch = 0
    trainer.global_step = 0

    trainer.load_checkpoint(checkpoint_path)

    _assert_nested_equal(trainer.optimizer.state_dict(), original_optimizer_state)
    _assert_nested_equal(trainer.scheduler.state_dict(), original_scheduler_state)
    assert trainer.get_state()["lr"] == [5e-4]


def test_trainer_save_and_load_checkpoint_round_trip(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    FakeTqdm.instances.clear()
    FakeTqdm.writes.clear()
    _patch_split_cache(
        monkeypatch,
        tmp_path,
        {
            "train": _make_block_splits(4),
            "validation": _make_block_splits(4),
            "test": _make_block_splits(4),
        },
    )
    monkeypatch.setattr("cr_train.data.runtime.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CountingStepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        metrics={"mae": mae_metric},
        scheduler=scheduler,
        max_train_samples=2 * BLOCK_SIZE,
        max_val_samples=BLOCK_SIZE,
        max_test_samples=BLOCK_SIZE,
        epochs=1,
        batch_size=8,
        seed=11,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    trainer.step()
    original_model_state = _clone_model_state_dict(model)
    original_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    original_scheduler_state = copy.deepcopy(trainer.scheduler.state_dict())
    checkpoint_path = trainer.save_checkpoint()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    for state in trainer.optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                value.zero_()
    trainer.scheduler.step()
    trainer.current_epoch = 0
    trainer.global_step = 0

    load_summary = trainer.load_checkpoint(checkpoint_path)
    metrics_records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert checkpoint_path == output_dir / "epoch-0001.pt"
    assert checkpoint_path.exists()
    assert load_summary == {
        "path": checkpoint_path,
        "epoch": 1,
        "global_step": (2 * BLOCK_SIZE) // 8,
    }
    assert trainer.current_epoch == 1
    assert trainer.global_step == (2 * BLOCK_SIZE) // 8
    _assert_nested_equal(_clone_model_state_dict(model), original_model_state)
    _assert_nested_equal(trainer.optimizer.state_dict(), original_optimizer_state)
    _assert_nested_equal(trainer.scheduler.state_dict(), original_scheduler_state)
    assert [record["kind"] for record in metrics_records if record["kind"].startswith("checkpoint")] == [
        "checkpoint_save",
        "checkpoint_load",
    ]


def test_trainer_load_checkpoint_keeps_scheduler_state_when_legacy_checkpoint_has_no_scheduler(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    _patch_split_cache(
        monkeypatch,
        tmp_path,
        {
            "train": _make_block_splits(4),
            "validation": _make_block_splits(4),
            "test": _make_block_splits(4),
        },
    )
    monkeypatch.setattr("cr_train.trainer.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.data.runtime.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CountingStepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        scheduler=scheduler,
        max_train_samples=2 * BLOCK_SIZE,
        max_val_samples=BLOCK_SIZE,
        max_test_samples=BLOCK_SIZE,
        epochs=1,
        batch_size=8,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    trainer.step()
    legacy_checkpoint_path = output_dir / "legacy-checkpoint.pt"
    torch.save(
        {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "model": trainer._model_state_owner().state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
        },
        legacy_checkpoint_path,
    )

    trainer.scheduler.step()
    expected_scheduler_state = copy.deepcopy(trainer.scheduler.state_dict())

    trainer.load_checkpoint(legacy_checkpoint_path)

    _assert_nested_equal(trainer.scheduler.state_dict(), expected_scheduler_state)


def test_trainer_save_and_load_weights_preserves_runtime_state(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    _patch_split_cache(
        monkeypatch,
        tmp_path,
        {
            "train": _make_block_splits(4),
            "validation": _make_block_splits(4),
            "test": _make_block_splits(4),
        },
    )
    monkeypatch.setattr("cr_train.trainer.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.data.runtime.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        metrics={"mae": mae_metric},
        max_train_samples=2 * BLOCK_SIZE,
        max_val_samples=BLOCK_SIZE,
        max_test_samples=BLOCK_SIZE,
        epochs=1,
        batch_size=8,
        seed=13,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    trainer.step()
    weights_path = trainer.save_weights()
    original_model_state = _clone_model_state_dict(model)
    expected_epoch = trainer.current_epoch
    expected_global_step = trainer.global_step
    first_state_tensor = next(
        value
        for state in trainer.optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    first_state_tensor.fill_(123.0)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)

    trainer.load_weights(weights_path)

    assert weights_path == output_dir / "model-epoch-0001.pt"
    assert weights_path.exists()
    assert trainer.current_epoch == expected_epoch
    assert trainer.global_step == expected_global_step
    _assert_nested_equal(_clone_model_state_dict(model), original_model_state)
    torch.testing.assert_close(first_state_tensor, torch.full_like(first_state_tensor, 123.0))


def test_trainer_load_weights_from_checkpoint_preserves_scheduler_and_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    _patch_split_cache(
        monkeypatch,
        tmp_path,
        {
            "train": _make_block_splits(4),
            "validation": _make_block_splits(4),
            "test": _make_block_splits(4),
        },
    )
    monkeypatch.setattr("cr_train.trainer.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.data.runtime.tqdm", FakeTqdm)
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CountingStepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        scheduler=scheduler,
        max_train_samples=2 * BLOCK_SIZE,
        max_val_samples=BLOCK_SIZE,
        max_test_samples=BLOCK_SIZE,
        epochs=1,
        batch_size=8,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    trainer.step()
    checkpoint_path = trainer.save_checkpoint()
    original_model_state = _clone_model_state_dict(model)
    first_state_tensor = next(
        value
        for state in trainer.optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    first_state_tensor.fill_(123.0)

    trainer.scheduler.step()
    expected_scheduler_state = copy.deepcopy(trainer.scheduler.state_dict())
    expected_lr = trainer.get_state()["lr"]
    trainer.current_epoch = 11
    trainer.global_step = 29

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)

    trainer.load_weights(checkpoint_path)

    _assert_nested_equal(_clone_model_state_dict(model), original_model_state)
    _assert_nested_equal(trainer.scheduler.state_dict(), expected_scheduler_state)
    assert trainer.current_epoch == 11
    assert trainer.global_step == 29
    assert trainer.get_state()["lr"] == expected_lr
    torch.testing.assert_close(first_state_tensor, torch.full_like(first_state_tensor, 123.0))


def test_trainer_predict_preserves_training_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
    )
    trainer.model.train(True)

    batch = {
        "sar": torch.randn(2, 2, 8, 8),
        "cloudy": torch.randn(2, 13, 8, 8),
        "target": torch.randn(2, 13, 8, 8),
        "meta": {"scene": ["a", "b"]},
    }
    prediction = trainer.predict(batch)

    assert prediction.shape == (2, 13, 8, 8)
    assert prediction.requires_grad is False
    assert trainer.model.training is True


def test_trainer_get_state_reports_runtime_values(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        epochs=3,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
    )
    trainer.current_epoch = 2
    trainer.global_step = 17

    assert trainer.get_state() == {
        "epoch": 2,
        "epochs": 3,
        "global_step": 17,
        "lr": [1e-3],
        "device": torch.device("cpu"),
        "distributed": False,
    }


def test_trainer_rejects_optimizer_from_other_model() -> None:
    model = TinyModel()
    other_model = TinyModel()

    with pytest.raises(ValueError):
        Trainer(
            model,
            torch.optim.AdamW(other_model.parameters(), lr=1e-3),
            loss_fn,
        )


def test_trainer_rejects_scheduler_from_other_optimizer() -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    other_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError):
        Trainer(
            model,
            optimizer,
            loss_fn,
            scheduler=torch.optim.lr_scheduler.StepLR(other_optimizer, step_size=1),
        )


def test_trainer_rejects_scheduler_monitor_without_scheduler() -> None:
    model = TinyModel()

    with pytest.raises(ValueError):
        Trainer(
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-3),
            loss_fn,
            scheduler_monitor="val.loss",
        )


def test_trainer_rejects_scheduler_monitor_for_non_plateau_scheduler() -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError):
        Trainer(
            model,
            optimizer,
            loss_fn,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
            scheduler_monitor="val.loss",
        )


@pytest.mark.parametrize(
    "scheduler_monitor",
    ["train.loss", "test.loss", "val.metrics.unknown", "val.metrics.", "bogus"],
)
def test_trainer_rejects_invalid_reduce_lr_on_plateau_monitor(
    scheduler_monitor: str,
) -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError):
        Trainer(
            model,
            optimizer,
            loss_fn,
            metrics={"mae": mae_metric},
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
            scheduler_monitor=scheduler_monitor,
        )


def test_trainer_rejects_non_positive_train_crop_size() -> None:
    model = TinyModel()

    with pytest.raises(ValueError):
        Trainer(
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-3),
            loss_fn,
            train_crop_size=0,
        )


def test_trainer_defers_output_dir_creation_until_first_write(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cr_train.trainer.resolve_num_workers", lambda _value: 0)

    output_dir = tmp_path / "run"
    model = TinyModel()
    Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    assert not output_dir.exists()


def test_trainer_wraps_model_for_distributed_without_device_bootstrap_bug(monkeypatch, tmp_path: Path) -> None:
    import cr_train.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "is_distributed", lambda: True)
    monkeypatch.setattr(trainer_mod, "DDP", FakeDDP)
    monkeypatch.setattr(trainer_mod, "resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
    )

    assert isinstance(trainer.model, FakeDDP)
    assert trainer.model.module is model
    assert trainer.device.type == "cpu"


def test_trainer_checkpoint_apis_use_unwrapped_model_state_in_distributed_mode(monkeypatch, tmp_path: Path) -> None:
    import cr_train.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "is_distributed", lambda: True)
    monkeypatch.setattr(trainer_mod, "DDP", FakeDDP)
    monkeypatch.setattr(trainer_mod, "resolve_num_workers", lambda _value: 0)

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
    )
    trainer.current_epoch = 3
    trainer.global_step = 21

    with torch.no_grad():
        trainer.model.module.body.weight.fill_(0.25)
        trainer.model.module.body.bias.fill_(0.5)

    checkpoint_path = trainer.save_checkpoint(tmp_path / "manual-checkpoint.pt")
    weights_path = trainer.save_weights(tmp_path / "manual-weights.pt")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    weights_payload = torch.load(weights_path, map_location="cpu")

    assert "module.body.weight" not in checkpoint_payload["model"]
    assert "module.body.weight" not in weights_payload

    restored_model = TinyModel()
    restored_trainer = Trainer(
        restored_model,
        torch.optim.AdamW(restored_model.parameters(), lr=1e-3),
        loss_fn,
        output_dir=tmp_path / "restore-run",
        cache_dir=tmp_path / "cache-restore",
    )
    load_summary = restored_trainer.load_checkpoint(checkpoint_path)

    assert load_summary["epoch"] == 3
    assert load_summary["global_step"] == 21
    torch.testing.assert_close(restored_model.body.weight, model.body.weight)
    torch.testing.assert_close(restored_model.body.bias, model.body.bias)


def test_trainer_resolves_spawn_context_on_cuda_workers(monkeypatch, tmp_path: Path) -> None:
    import cr_train.trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod.Trainer,
        "_infer_module_device",
        staticmethod(lambda _module: torch.device("cuda:0")),
    )

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        num_workers=2,
    )

    assert trainer.num_workers == 2
    assert trainer.multiprocessing_context == "spawn"


def test_trainer_respects_explicit_multiprocessing_context(monkeypatch, tmp_path: Path) -> None:
    import cr_train.trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod.Trainer,
        "_infer_module_device",
        staticmethod(lambda _module: torch.device("cuda:0")),
    )

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        num_workers=2,
        multiprocessing_context="forkserver",
    )

    assert trainer.multiprocessing_context == "forkserver"


def test_trainer_disables_multiprocessing_context_when_workers_are_disabled(tmp_path: Path) -> None:
    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        num_workers=0,
        multiprocessing_context="spawn",
    )

    assert trainer.num_workers == 0
    assert trainer.multiprocessing_context is None


def test_trainer_reuses_prepared_split_state_across_epochs(monkeypatch, tmp_path: Path) -> None:
    import cr_train.data.dataset as dataset_mod
    import cr_train.trainer as trainer_mod

    FakeTqdm.instances.clear()
    FakeTqdm.writes.clear()
    _patch_split_cache(
        monkeypatch,
        tmp_path,
        {
            "train": _make_block_splits(4),
            "validation": _make_block_splits(4),
            "test": _make_block_splits(4),
        },
    )
    monkeypatch.setattr(trainer_mod, "tqdm", FakeTqdm)
    monkeypatch.setattr(trainer_mod, "resolve_num_workers", lambda _value: 0)

    resolve_counts: dict[str, int] = defaultdict(int)
    real_resolve = dataset_mod.resolve_prepared_split_state

    def counting_resolve(**kwargs):
        resolve_counts[str(kwargs["split"])] += 1
        return real_resolve(**kwargs)

    monkeypatch.setattr(trainer_mod, "resolve_prepared_split_state", counting_resolve)

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        metrics={"mae": mae_metric},
        max_train_samples=2 * BLOCK_SIZE,
        max_val_samples=BLOCK_SIZE,
        max_test_samples=BLOCK_SIZE,
        epochs=2,
        batch_size=8,
        seed=7,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
    )

    trainer.step()
    trainer.step()
    trainer.test()

    assert trainer.persistent_workers is False
    assert resolve_counts == {"train": 1, "validation": 1, "test": 1}


def test_trainer_passes_spatial_transform_options_only_to_train_loader(monkeypatch, tmp_path: Path) -> None:
    import cr_train.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "resolve_num_workers", lambda _value: 0)

    captured: list[dict[str, object]] = []

    def fake_run_startup_stage(_startup_callback, **kwargs):
        return kwargs["operation"]()

    def fake_build_dataloader(prepared, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(dataset=prepared.dataset, sampler=None)

    monkeypatch.setattr(trainer_mod, "run_startup_stage", fake_run_startup_stage)
    monkeypatch.setattr(trainer_mod, "build_dataloader", fake_build_dataloader)
    monkeypatch.setattr(
        trainer_mod,
        "prepare_split_from_state",
        lambda _state, **_kwargs: SimpleNamespace(
            dataset=SimpleNamespace(name="dataset"),
            num_examples=8,
        ),
    )

    model = TinyModel()
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        loss_fn,
        batch_size=4,
        output_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        train_crop_size=128,
        train_random_flip=True,
        train_random_rot90=True,
    )
    trainer._resolve_prepared_split_state = lambda *, split, max_samples: SimpleNamespace(split=split, max_samples=max_samples)

    trainer._build_loader(
        split="train",
        max_samples=BLOCK_SIZE,
        training=True,
        epoch_index=0,
    )
    trainer._build_loader(
        split="validation",
        max_samples=BLOCK_SIZE,
        training=False,
        epoch_index=0,
    )

    assert captured[0]["crop_size"] == 128
    assert captured[0]["crop_mode"] == "random"
    assert captured[0]["multiprocessing_context"] is None
    assert captured[0]["random_flip"] is True
    assert captured[0]["random_rot90"] is True
    assert captured[1]["crop_size"] is None
    assert captured[1]["crop_mode"] == "none"
    assert captured[1]["multiprocessing_context"] is None
    assert captured[1]["random_flip"] is False
    assert captured[1]["random_rot90"] is False
