from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import cr_train.data as data_mod
from cr_train.data import (
    DataModuleConfig,
    LoaderConfig,
    SEN12MSCRDataModule,
    SceneShard,
    ShuffleConfig,
    SplitRatios,
    decode_sample,
    official_scene_splits,
    seeded_scene_splits,
)


def _sample_row(*, season: str = "spring", scene: str = "1", patch: str = "p30") -> dict[str, object]:
    sar = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    optical = np.arange(78, dtype=np.int16).reshape(3, 2, 13)
    return {
        "sar": sar.tobytes(),
        "cloudy": optical.tobytes(),
        "target": (optical + 1).tobytes(),
        "sar_shape": list(sar.shape),
        "opt_shape": list(optical.shape),
        "dtype": "float32",
        "season": season,
        "scene": scene,
        "patch": patch,
    }


def test_decode_sample_transforms_bytes_into_tensors() -> None:
    decoded = decode_sample(_sample_row(), tensor_layout="channels_first")

    assert decoded["sar"].shape == (2, 3, 2)
    assert decoded["cloudy"].shape == (13, 3, 2)
    assert decoded["target"].shape == (13, 3, 2)
    assert decoded["metadata"]["source_shard"] == "spring/scene_1.parquet"
    assert decoded["metadata"]["patch"] == "p30"
    assert decoded["sar"].dtype == torch.float32
    assert decoded["cloudy"].dtype == torch.int16


def test_official_scene_splits_are_scene_isolated() -> None:
    splits = official_scene_splits()
    assert len(splits["train"]) == 155
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10

    train_ids = {scene.source_id for scene in splits["train"]}
    val_ids = {scene.source_id for scene in splits["val"]}
    test_ids = {scene.source_id for scene in splits["test"]}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_seeded_scene_splits_are_deterministic_and_disjoint() -> None:
    catalog = tuple(
        SceneShard(season=season, scene=str(index))
        for season in ("spring", "summer")
        for index in range(1, 7)
    )
    first = seeded_scene_splits(seed=7, split_ratios=SplitRatios(0.5, 0.25, 0.25), scene_catalog=catalog)
    second = seeded_scene_splits(seed=7, split_ratios=SplitRatios(0.5, 0.25, 0.25), scene_catalog=catalog)

    assert first == second
    assert sum(len(shards) for shards in first.values()) == len(catalog)
    assert {scene.source_id for scene in first["train"]}.isdisjoint(
        {scene.source_id for scene in first["val"]}
    )


class _FakeIterable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[object, ...]] = []

    def reshard(self, num_shards: int) -> "_FakeIterable":
        self.calls.append(("reshard", num_shards))
        return self

    def shuffle(self, *, seed: int, buffer_size: int) -> "_FakeIterable":
        self.calls.append(("shuffle", seed, buffer_size))
        return self

    def set_epoch(self, epoch: int) -> None:
        self.calls.append(("set_epoch", epoch))

    def __iter__(self):
        return iter(self.rows)


def test_train_pipeline_reshards_before_shuffle() -> None:
    fake_iterable = _FakeIterable([_sample_row()])
    datamodule = SEN12MSCRDataModule(
        DataModuleConfig(
            seed=13,
            loader=LoaderConfig(batch_size=1),
            shuffle=ShuffleConfig(enabled=True, buffer_size=16, reshard_num_shards=8),
        ),
        dataset_loader=lambda urls, stage: fake_iterable,
        scene_split_resolver=lambda _: {
            "train": (SceneShard("spring", "1"),),
            "val": (),
            "test": (),
        },
    )

    batch = next(iter(datamodule.train_dataloader(epoch=3)))

    assert tuple(batch["sar"].shape) == (1, 2, 3, 2)
    assert fake_iterable.calls[:3] == [
        ("reshard", 8),
        ("shuffle", 13, 16),
        ("set_epoch", 3),
    ]


def test_loader_config_rejects_invalid_worker_settings() -> None:
    with pytest.raises(ValueError):
        LoaderConfig(persistent_workers=True, num_workers=0)


def test_runtime_is_not_configured_on_import() -> None:
    runtime_mod = importlib.import_module("cr_train.runtime")
    runtime_mod = importlib.reload(runtime_mod)

    assert runtime_mod._CONFIGURED is False


def test_default_dataset_loader_bootstraps_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    dummy_iterable = _FakeIterable([])

    def fake_configure_runtime() -> None:
        calls.append("configure")

    def fake_load_dataset(*args: object, **kwargs: object) -> _FakeIterable:
        calls.append(("load_dataset", kwargs["data_files"], kwargs["split"], kwargs["streaming"]))
        return dummy_iterable

    monkeypatch.setattr(data_mod, "configure_runtime", fake_configure_runtime)
    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)

    dataset = data_mod._default_dataset_loader(["/tmp/sample.parquet"], "train")

    assert dataset is dummy_iterable
    assert calls == [
        "configure",
        ("load_dataset", {"train": ["/tmp/sample.parquet"]}, "train", True),
    ]


def test_runtime_patch_allows_clean_subprocess_exit(tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    table = pa.table({"value": [1]})
    pq.write_table(table, parquet_path)

    script = textwrap.dedent(
        """
        from cr_train.data import _default_dataset_loader
        import sys

        dataset = _default_dataset_loader([sys.argv[1]], "train")
        row = next(iter(dataset))
        print(row["value"])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(parquet_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1"
