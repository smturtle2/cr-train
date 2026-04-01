from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from cr_train import Trainer
from cr_train.data import BLOCK_SIZE


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


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Conv2d(15, 13, kernel_size=1)

    def forward(self, sar, cloudy):
        return self.body(torch.cat([sar, cloudy], dim=1))


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
        self.postfixes: list[dict[str, object]] = []
        self.desc_history: list[str] = [str(self.desc)] if self.desc is not None else []
        FakeTqdm.instances.append(self)

    @staticmethod
    def write(message: str) -> None:
        FakeTqdm.writes.append(message)

    def update(self, value: int) -> None:
        self.updates.append(value)

    def set_postfix(self, values: dict[str, object]) -> None:
        self.postfixes.append(values)

    def set_description_str(self, desc: str, refresh: bool = True) -> None:
        del refresh
        self.desc = desc
        self.desc_history.append(desc)

    def close(self) -> None:
        return None


class _FakeStreamingDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = [dict(row) for row in rows]

    def select_columns(self, columns: list[str]):
        return _FakeStreamingDataset(
            [{column: row[column] for column in columns} for row in self._rows]
        )

    def shuffle(self, *, seed: int, buffer_size: int):
        del buffer_size
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self._rows))
        return _FakeStreamingDataset([self._rows[int(index)] for index in order])

    def __iter__(self):
        for row in self._rows:
            yield dict(row)


class _FakeRowGroupMetadata:
    def __init__(self, num_rows: int) -> None:
        self.num_rows = num_rows


class _FakeParquetMetadata:
    def __init__(self, row_group_rows: list[int]) -> None:
        self.num_rows = sum(row_group_rows)
        self.num_row_groups = len(row_group_rows)
        self._row_group_rows = row_group_rows

    def row_group(self, index: int) -> _FakeRowGroupMetadata:
        return _FakeRowGroupMetadata(self._row_group_rows[index])


class _FakeParquetFile:
    def __init__(
        self,
        url: str,
        row_group_rows: list[int],
        stats: dict[str, object],
    ) -> None:
        self.metadata = _FakeParquetMetadata(row_group_rows)
        stats["opens"][url] += 1


def _patch_source(monkeypatch, split_rows: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    from cr_train.data.source import _source_descriptor_cache
    _source_descriptor_cache.clear()

    parquet_entries: list[dict[str, object]] = []
    shard_rows: dict[str, list[int]] = {}
    stats: dict[str, object] = {
        "opens": defaultdict(int),
        "load_dataset_calls": [],
        "request_json_calls": 0,
    }

    for split, rows in split_rows.items():
        url = f"hf://datasets/unit/test@refs/convert/parquet/default/{split}/0000.parquet"
        parquet_entries.append(
            {
                "dataset": "unit/test",
                "config": "default",
                "split": split,
                "url": url,
                "filename": "0000.parquet",
            }
        )
        shard_rows[url] = [
            min(BLOCK_SIZE, len(rows) - start)
            for start in range(0, len(rows), BLOCK_SIZE)
        ]

    def fake_request_json(url: str):
        stats["request_json_calls"] += 1
        if "/info?" in url:
            return {
                "dataset_info": {
                    "default": {
                        "splits": {
                            split: {"num_examples": len(rows)}
                            for split, rows in split_rows.items()
                        }
                    }
                }
            }
        if "/parquet?" in url:
            return {"parquet_files": parquet_entries}
        raise AssertionError(f"unexpected URL: {url}")

    def fake_parquet_file(url: str):
        return _FakeParquetFile(str(url), row_group_rows=list(shard_rows[str(url)]), stats=stats)

    def fake_load_dataset(path: str, *args, split: str | None = None, streaming: bool = False, **kwargs):
        del args, kwargs
        assert path == "Hermanni/sen12mscr"
        assert split is not None
        assert streaming is True
        stats["load_dataset_calls"].append(str(split))
        return _FakeStreamingDataset(list(split_rows[str(split)]))

    monkeypatch.setattr("cr_train.data.source.request_json", fake_request_json)
    monkeypatch.setattr("cr_train.data.source.pq.ParquetFile", fake_parquet_file)
    monkeypatch.setattr("cr_train.data.runtime.load_dataset", fake_load_dataset)
    return stats


def test_trainer_step_and_test_with_block_cache_warmup(monkeypatch, tmp_path: Path) -> None:
    rows = [_make_row(i) for i in range(4 * BLOCK_SIZE)]
    output_dir = tmp_path / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    metrics_path.write_text("old-record\n", encoding="utf-8")

    FakeTqdm.instances.clear()
    FakeTqdm.writes.clear()
    stats = _patch_source(
        monkeypatch,
        {
            "train": rows,
            "validation": rows,
            "test": rows,
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
        dataset_seed=23,
        output_dir=output_dir,
        cache_dir=tmp_path / "cache",
    )

    assert metrics_path.read_text(encoding="utf-8") == ""

    epoch_summary = trainer.step()
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
    warmup_bars = [
        instance
        for instance in FakeTqdm.instances
        if any("cache " in desc for desc in instance.desc_history)
    ]
    config_record = next(record for record in metrics_records if record["kind"] == "config")

    assert epoch_summary["epoch"] == 1
    assert epoch_summary["train"]["loss"] >= 0.0
    assert "mae" in epoch_summary["train"]["metrics"]
    assert epoch_summary["checkpoint_path"].endswith("epoch-0001.pt")
    assert Path(epoch_summary["checkpoint_path"]).exists()

    assert test_summary["epoch"] == 1
    assert "mae" in test_summary["metrics"]
    assert epoch_summary["train"]["num_batches"] == 4
    assert "batches_per_sec" in epoch_summary["train"]
    assert "old-record" not in metrics_path.read_text(encoding="utf-8")

    assert config_record["dataset_seed"] == 23
    assert "ensure catalog" in startup_stages
    assert "warm split cache" in startup_stages
    assert "load local cache" in startup_stages
    assert "build dataloader" in startup_stages
    assert "wait first batch" in startup_stages
    assert "start epoch" in startup_stages

    # writes: 1 config banner + 3 cache summaries + 1 epoch summary + 1 test summary = 6
    assert len(FakeTqdm.writes) == 6
    assert not any(message.startswith("loader ") for message in FakeTqdm.writes)
    assert not any(message.startswith("train | epoch=") for message in FakeTqdm.writes)
    assert any("cr-train" in message for message in FakeTqdm.writes)
    assert any("Epoch 1/" in message for message in FakeTqdm.writes)
    assert any("Test" in message for message in FakeTqdm.writes)

    assert len(warmup_bars) == 3
    assert all(int(bar.total) >= 1 for bar in warmup_bars)
    assert all(sum(bar.updates) == int(bar.total) for bar in warmup_bars)
    assert all(any("fill" in values for values in bar.postfixes) for bar in warmup_bars)
    assert all(any("sel" in values for values in bar.postfixes) for bar in warmup_bars)
    assert all(any("blk/s" in values for values in bar.postfixes) for bar in warmup_bars)
    assert all(all("runs" not in values for values in bar.postfixes) for bar in warmup_bars)

    assert len(batch_bars) == 3
    assert any(any("loss" in values and "mae" in values for values in bar.postfixes) for bar in batch_bars)
    assert any(any("batches/s" in values for values in bar.postfixes) for bar in batch_bars)
    assert all(all(value == 1 for value in bar.updates) for bar in batch_bars)

    warmup_done_records = [
        record
        for record in startup_records
        if record["stage"] == "warm split cache" and record["status"] == "done"
    ]
    warmup_records_by_split = {record["split"]: record for record in warmup_done_records[:3]}
    warmup_bars_by_split = {}
    for bar in warmup_bars:
        for split in ("train", "validation", "test"):
            if any(desc.endswith(f"cache {split}") for desc in bar.desc_history):
                warmup_bars_by_split[split] = bar
                break

    assert [record["selected_block_count"] for record in warmup_done_records[:3]] == [2, 1, 1]
    assert all(record["planner_mode"] == "stop_biased_exact_k" for record in warmup_done_records[:3])
    assert all(record["stop_bias_alpha"] == 8.0 for record in warmup_done_records[:3])
    assert all("timeline" in record for record in warmup_done_records[:3])
    assert all(len(str(record["timeline"])) == int(record["execution_block_count"]) for record in warmup_done_records[:3])
    assert all(set(str(record["timeline"])) <= {"█", "░"} for record in warmup_done_records[:3])
    assert all(record["frontier_before"] == 0 for record in warmup_done_records[:3])
    assert all(record["frontier_after"] == record["execution_block_count"] for record in warmup_done_records[:3])
    assert all(record["extension_blocks"] == record["execution_block_count"] for record in warmup_done_records[:3])

    for split, record in warmup_records_by_split.items():
        expected_summary = (
            f"{record['timeline']} cache {split} | warm | "
            f"selected={record['selected_block_count']} frontier={record['frontier_before']}->{record['frontier_after']}"
        )
        assert expected_summary in FakeTqdm.writes

    assert stats["load_dataset_calls"] == ["train", "validation", "test"]
    assert stats["request_json_calls"] == 2

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
