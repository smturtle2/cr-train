from __future__ import annotations

import importlib
import json
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cr_train import Trainer
from cr_train.data import BLOCK_SIZE
from cr_train.data.store import resolve_block_cache_paths
from cr_train.trainer_reporting import format_cache_summary, format_epoch_summary


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


def loss_fn(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.nn.functional.mse_loss(prediction, batch["target"])


def mae_metric(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - batch["target"]))


class FakeTqdm:
    instances: list["FakeTqdm"] = []
    writes: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.disable = kwargs.get("disable", False)
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

    assert "\n" not in summary
    assert "1.2s" in summary
    assert "■□" in summary
    assert "█" not in summary and "░" not in summary
    assert "…" in summary


def test_format_epoch_summary_prefers_elapsed_time_over_throughput() -> None:
    summary = format_epoch_summary(
        {
            "epoch": 1,
            "train": {
                "loss": 0.0423,
                "metrics": {"mae": 0.0312},
                "samples_per_sec": 142.3,
            },
            "val": {
                "loss": 0.0391,
                "metrics": {"mae": 0.0298},
            },
            "elapsed_sec": 12.34,
        },
        epochs=2,
    )

    assert "12.3s" in summary
    assert "samples/s" not in summary


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
    checkpoint_record = next(record for record in metrics_records if record["kind"] == "checkpoint")

    batch_bars = [
        instance
        for instance in FakeTqdm.instances
        if str(instance.desc).startswith(("train", "val", "test"))
    ]
    warmup_bars = [
        instance
        for instance in FakeTqdm.instances
        if any("cache " in desc for desc in instance.desc_history)
    ]
    config_record = next(record for record in metrics_records if record["kind"] == "config")

    assert epoch_summary["epoch"] == 1
    assert epoch_summary["elapsed_sec"] > 0.0
    assert epoch_summary["train"]["loss"] >= 0.0
    assert "mae" in epoch_summary["train"]["metrics"]
    assert epoch_summary["checkpoint_path"].endswith("epoch-0001.pt")
    assert Path(epoch_summary["checkpoint_path"]).exists()

    assert test_summary["epoch"] == 1
    assert "mae" in test_summary["metrics"]
    assert epoch_summary["train"]["num_batches"] == (2 * BLOCK_SIZE) // 8
    assert "batches_per_sec" in epoch_summary["train"]
    assert checkpoint_record["elapsed_sec"] > 0.0
    assert "old-record" not in metrics_path.read_text(encoding="utf-8")

    assert "dataset_seed" not in config_record
    assert config_record["train_crop_size"] is None
    assert config_record["train_random_flip"] is False
    assert config_record["train_random_rot90"] is False
    assert warmup_splits_after_step == {"train", "validation", "test"}
    assert "warm split cache" in startup_stages
    assert "load local cache" in startup_stages
    assert "build dataloader" in startup_stages
    assert "wait first batch" in startup_stages
    assert "start epoch" in startup_stages

    assert len(FakeTqdm.writes) >= 6
    assert any("cr-train" in message for message in FakeTqdm.writes)
    assert any("Epoch 1/" in message for message in FakeTqdm.writes)
    assert any("Test" in message for message in FakeTqdm.writes)
    epoch_message = next(message for message in FakeTqdm.writes if "Epoch 1/" in message)
    assert "samples/s" not in epoch_message
    assert re.search(r"\d+\.\d+s", epoch_message)
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

    assert len(batch_bars) == 3
    assert any(any("loss" in values and "mae" in values for values in bar.postfixes) for bar in batch_bars)
    assert all(all("batches/s" not in values for values in bar.postfixes) for bar in batch_bars)
    assert all(all(value == 1 for value in bar.updates) for bar in batch_bars)

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


def test_trainer_rejects_optimizer_from_other_model() -> None:
    model = TinyModel()
    other_model = TinyModel()

    with pytest.raises(ValueError):
        Trainer(
            model,
            torch.optim.AdamW(other_model.parameters(), lr=1e-3),
            loss_fn,
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
    assert captured[0]["random_flip"] is True
    assert captured[0]["random_rot90"] is True
    assert captured[1]["crop_size"] is None
    assert captured[1]["crop_mode"] == "none"
    assert captured[1]["random_flip"] is False
    assert captured[1]["random_rot90"] is False
