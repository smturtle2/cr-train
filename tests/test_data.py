from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import cr_train.data as data_mod
from cr_train import build_sen12mscr_loaders
from cr_train.data import (
    DEFAULT_DATASET_REVISION,
    DEFAULT_SHUFFLE_BUFFER_SIZE,
    HFTokenStatus,
    PARQUET_COLUMNS,
    SceneShard,
    decode_sample,
    hf_token_configured,
    hf_token_status,
    official_scene_splits,
    seeded_scene_splits,
)


def _sample_row(*, season: str = "spring", scene: str = "1", patch: str = "p30") -> dict[str, object]:
    sar = np.array(
        [
            [[-30.0, -5.0], [-12.5, 1.0]],
            [[-25.0, -7.5], [0.0, -20.0]],
        ],
        dtype=np.float32,
    )
    optical = np.arange(13 * 2 * 1, dtype=np.int16).reshape(2, 1, 13) * 800 - 10
    return {
        "sar": sar.tobytes(),
        "cloudy": optical.tobytes(),
        "target": (optical + 100).tobytes(),
        "sar_shape": list(sar.shape),
        "opt_shape": list(optical.shape),
        "season": season,
        "scene": scene,
        "patch": patch,
    }


def _fake_scene_splits(_: str, __: int) -> dict[str, tuple[SceneShard, ...]]:
    return {
        "train": (SceneShard("spring", "1"),),
        "val": (SceneShard("spring", "2"),),
        "test": (SceneShard("spring", "3"),),
    }


class _FakeIterable(torch.utils.data.IterableDataset[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        super().__init__()
        self.rows = rows
        self.calls: list[str] = []

    def __iter__(self):
        self.calls.append("iter")
        return iter(self.rows)


class _FakeHFSource(torch.utils.data.IterableDataset[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        super().__init__()
        self.rows = rows
        self.shuffle_calls: list[tuple[int, int]] = []
        self.epochs: list[int] = []

    def shuffle(self, *, seed: int, buffer_size: int) -> "_FakeHFSource":
        self.shuffle_calls.append((seed, buffer_size))
        return self

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)

    def __iter__(self):
        return iter(self.rows)


def test_decode_sample_applies_official_preprocessing_and_standard_schema() -> None:
    decoded = decode_sample(_sample_row())

    sar, cloudy = decoded["inputs"]
    assert sar.shape == (2, 2, 2)
    assert cloudy.shape == (13, 2, 1)
    assert decoded["target"].shape == (13, 2, 1)
    assert decoded["metadata"]["source_shard"] == "spring/scene_1.parquet"
    assert decoded["metadata"]["patch"] == "p30"
    assert sar.dtype == torch.float32
    assert cloudy.dtype == torch.float32
    assert decoded["target"].dtype == torch.float32
    assert torch.isclose(sar.min(), torch.tensor(0.0))
    assert torch.isclose(sar.max(), torch.tensor(1.0))
    assert torch.isclose(cloudy.min(), torch.tensor(0.0))
    assert torch.isclose(cloudy.max(), torch.tensor(1.0))
    assert torch.isclose(decoded["target"].max(), torch.tensor(1.0))


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
    first = seeded_scene_splits(seed=7, scene_catalog=catalog)
    second = seeded_scene_splits(seed=7, scene_catalog=catalog)

    assert first == second
    assert sum(len(shards) for shards in first.values()) == len(catalog)
    assert {scene.source_id for scene in first["train"]}.isdisjoint(
        {scene.source_id for scene in first["val"]}
    )


def test_scene_shard_urls_are_pinned_to_a_dataset_revision() -> None:
    url = SceneShard("spring", "1").resolve_url()
    assert f"@{DEFAULT_DATASET_REVISION}/" in url


def test_hf_token_configured_reflects_huggingface_auth_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(data_mod, "get_token", lambda: None)
    assert hf_token_configured() is False

    monkeypatch.setattr(data_mod, "get_token", lambda: "hf_test_token")
    assert hf_token_configured() is True


def test_hf_token_status_reports_env_and_cached_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")
    monkeypatch.setattr(data_mod, "get_token", lambda: "hf_env_token")
    assert hf_token_status() == HFTokenStatus(
        configured=True,
        source="env",
        applied_to_hf=True,
    )

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(data_mod, "get_token", lambda: "hf_cached_token")
    assert hf_token_status() == HFTokenStatus(
        configured=True,
        source="cached",
        applied_to_hf=True,
    )

    monkeypatch.setattr(data_mod, "get_token", lambda: None)
    assert hf_token_status() == HFTokenStatus(
        configured=False,
        source="none",
        applied_to_hf=False,
    )


def test_build_loaders_defaults_to_official_split_and_decodes_samples() -> None:
    resolver_calls: list[tuple[str, int]] = []
    datasets = {
        "train": _FakeIterable([_sample_row()]),
        "val": _FakeIterable([_sample_row(scene="2")]),
        "test": _FakeIterable([_sample_row(scene="3")]),
    }

    def fake_dataset_loader(urls: list[str], stage: str) -> _FakeIterable:
        _ = urls
        return datasets[stage]

    def fake_scene_split_resolver(split: str, seed: int) -> dict[str, tuple[SceneShard, ...]]:
        resolver_calls.append((split, seed))
        return _fake_scene_splits(split, seed)

    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        1,
        seed=13,
        num_workers=0,
        _dataset_loader=fake_dataset_loader,
        _scene_split_resolver=fake_scene_split_resolver,
    )

    batch = next(iter(train_loader))
    _ = next(iter(val_loader))
    _ = next(iter(test_loader))

    assert resolver_calls == [("official", 13)]
    assert datasets["train"].calls == ["iter"]
    assert datasets["val"].calls == ["iter"]
    assert datasets["test"].calls == ["iter"]

    sar, _ = batch["inputs"]
    assert tuple(sar.shape) == (1, 2, 2, 2)
    assert tuple(batch["target"].shape) == (1, 13, 2, 1)


def test_build_loaders_supports_seeded_scene_split_without_custom_resolver() -> None:
    created: list[tuple[str, int]] = []
    datasets = {
        "train": _FakeIterable([_sample_row()]),
        "val": _FakeIterable([_sample_row(scene="2")]),
        "test": _FakeIterable([_sample_row(scene="3")]),
    }
    expected_splits = seeded_scene_splits(seed=7)

    def fake_dataset_loader(urls: list[str], stage: str) -> _FakeIterable:
        created.append((stage, len(urls)))
        return datasets[stage]

    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        1,
        seed=7,
        split="seeded_scene",
        num_workers=0,
        _dataset_loader=fake_dataset_loader,
    )

    _ = next(iter(train_loader))
    _ = next(iter(val_loader))
    _ = next(iter(test_loader))

    assert created == [(stage, len(expected_splits[stage])) for stage in ("train", "val", "test")]
    assert datasets["train"].calls == ["iter"]
    assert datasets["val"].calls == ["iter"]
    assert datasets["test"].calls == ["iter"]


def test_loader_builder_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(0)
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, split="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, num_workers=-1)
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, num_workers=0, timeout=1)
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, num_workers=2, prefetch_factor=0)
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, num_workers=0, prefetch_factor=2)
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, num_workers=0, persistent_workers=True)


def test_loader_builder_auto_tunes_simple_worker_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []
    datasets = {
        "train": _FakeIterable([_sample_row()]),
        "val": _FakeIterable([_sample_row(scene="2")]),
        "test": _FakeIterable([_sample_row(scene="3")]),
    }

    class _FakeDataLoader:
        def __init__(self, **kwargs: object) -> None:
            created.append(dict(kwargs))
            self.dataset = kwargs["dataset"]

        def __class_getitem__(cls, _: object) -> type["_FakeDataLoader"]:
            return cls

    def fake_dataset_loader(urls: list[str], stage: str) -> _FakeIterable:
        _ = urls
        return datasets[stage]

    monkeypatch.setattr(data_mod, "DataLoader", _FakeDataLoader)
    monkeypatch.setattr(data_mod.os, "cpu_count", lambda: 12)

    build_sen12mscr_loaders(
        2,
        _dataset_loader=fake_dataset_loader,
        _scene_split_resolver=_fake_scene_splits,
    )

    assert [kwargs["num_workers"] for kwargs in created] == [3, 1, 1]
    for kwargs in created:
        assert kwargs["batch_size"] == 2
        assert kwargs["worker_init_fn"] is data_mod._seed_worker
        assert isinstance(kwargs["generator"], torch.Generator)
        if kwargs["num_workers"] > 0:
            assert kwargs["timeout"] == 0.0
            assert kwargs["prefetch_factor"] == 2
            assert kwargs["persistent_workers"] is False
        else:
            assert kwargs["timeout"] == 0.0
            assert "prefetch_factor" not in kwargs
            assert "persistent_workers" not in kwargs


def test_loader_builder_auto_workers_do_not_exceed_streaming_source_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    class _FakeDataLoader:
        def __init__(self, **kwargs: object) -> None:
            created.append(dict(kwargs))
            self.dataset = kwargs["dataset"]

        def __class_getitem__(cls, _: object) -> type["_FakeDataLoader"]:
            return cls

    def fake_load_dataset(*args: object, **kwargs: object) -> _FakeHFSource:
        _ = (args, kwargs)
        source = _FakeHFSource([_sample_row()])
        source.num_shards = 2
        return source

    def scene_splits(_: str, __: int) -> dict[str, tuple[SceneShard, ...]]:
        return {
            "train": (SceneShard("spring", "1"), SceneShard("summer", "2")),
            "val": (SceneShard("fall", "3"),),
            "test": (SceneShard("winter", "4"),),
        }

    monkeypatch.setattr(data_mod, "DataLoader", _FakeDataLoader)
    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(data_mod.os, "cpu_count", lambda: 64)

    build_sen12mscr_loaders(
        2,
        _scene_split_resolver=scene_splits,
    )

    assert [kwargs["num_workers"] for kwargs in created] == [2, 1, 1]


def test_loader_builder_passes_explicit_dataloader_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []
    datasets = {
        "train": _FakeIterable([_sample_row()]),
        "val": _FakeIterable([_sample_row(scene="2")]),
        "test": _FakeIterable([_sample_row(scene="3")]),
    }

    class _FakeDataLoader:
        def __init__(self, **kwargs: object) -> None:
            created.append(dict(kwargs))
            self.dataset = kwargs["dataset"]

        def __class_getitem__(cls, _: object) -> type["_FakeDataLoader"]:
            return cls

    def fake_dataset_loader(urls: list[str], stage: str) -> _FakeIterable:
        _ = urls
        return datasets[stage]

    monkeypatch.setattr(data_mod, "DataLoader", _FakeDataLoader)

    build_sen12mscr_loaders(
        2,
        num_workers=2,
        pin_memory=True,
        timeout=5.5,
        prefetch_factor=4,
        persistent_workers=True,
        _dataset_loader=fake_dataset_loader,
        _scene_split_resolver=_fake_scene_splits,
    )

    assert len(created) == 3
    for kwargs in created:
        assert kwargs["batch_size"] == 2
        assert kwargs["num_workers"] == 2
        assert kwargs["pin_memory"] is True
        assert kwargs["worker_init_fn"] is data_mod._seed_worker
        assert isinstance(kwargs["generator"], torch.Generator)
        assert kwargs["timeout"] == 5.5
        assert kwargs["prefetch_factor"] == 4
        assert kwargs["persistent_workers"] is True


def test_collate_sen12mscr_rows_decodes_raw_batches() -> None:
    batch = data_mod._collate_sen12mscr_rows([
        _sample_row(patch="p0"),
        _sample_row(patch="p1"),
    ])

    sar, cloudy = batch["inputs"]
    assert tuple(sar.shape) == (2, 2, 2, 2)
    assert tuple(cloudy.shape) == (2, 13, 2, 1)
    assert tuple(batch["target"].shape) == (2, 13, 2, 1)
    assert batch["metadata"]["patch"] == ["p0", "p1"]


def test_seed_worker_keeps_numpy_seed_in_uint32_range(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(data_mod.torch, "initial_seed", lambda: (2**32) - 1)
    monkeypatch.setattr(data_mod.random, "seed", lambda value: calls.append(("python", value)))
    monkeypatch.setattr(data_mod.np.random, "seed", lambda value: calls.append(("numpy", value)))

    data_mod._seed_worker(1)

    assert calls == [("python", (2**32) - 1), ("numpy", (2**32) - 1)]


def test_collate_sen12mscr_rows_preserves_predecoded_batches() -> None:
    sample = decode_sample(_sample_row(patch="p1"))
    batch = data_mod._collate_sen12mscr_rows([sample])

    assert batch["metadata"]["patch"] == ["p1"]
    assert tuple(batch["target"].shape) == (1, 13, 2, 1)


def test_train_loader_keeps_hf_set_epoch_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[_FakeHFSource] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> _FakeHFSource:
        _ = (args, kwargs)
        source = _FakeHFSource([_sample_row()])
        created.append(source)
        return source

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)

    train_loader, _, _ = build_sen12mscr_loaders(
        1,
        seed=13,
        num_workers=0,
        _scene_split_resolver=_fake_scene_splits,
    )

    train_loader.dataset.set_epoch(3)
    _ = next(iter(train_loader))
    train_loader.dataset.set_epoch(7)
    _ = next(iter(train_loader))

    assert [source.epochs for source in created] == [[3], [7]]


def test_default_hf_streaming_loader_uses_shuffle_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    sources: list[_FakeHFSource] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> _FakeHFSource:
        _ = args
        call = dict(kwargs)
        calls.append(call)
        url = call["data_files"]["train"][0]
        source = _FakeHFSource([_sample_row(scene=url.rsplit("_", 1)[-1].split(".")[0])])
        sources.append(source)
        return source

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(data_mod, "get_token", lambda: "hf_test_token")

    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        1,
        seed=13,
        num_workers=0,
        _scene_split_resolver=_fake_scene_splits,
    )

    train_loader.dataset.set_epoch(5)
    batch = next(iter(train_loader))
    _ = next(iter(val_loader))
    _ = next(iter(test_loader))

    assert len(calls) == 3
    for call in calls:
        assert call["streaming"] is True
        assert call["split"] == "train"
        assert call["columns"] == list(PARQUET_COLUMNS)
        assert call["token"] == "hf_test_token"
        assert "batch_size" not in call

    assert sources[0].shuffle_calls == [(13, DEFAULT_SHUFFLE_BUFFER_SIZE)]
    assert sources[0].epochs == [5]
    assert sources[1].shuffle_calls == []
    assert sources[2].shuffle_calls == []
    assert tuple(batch["target"].shape) == (1, 13, 2, 1)


def test_default_hf_streaming_loader_rebuilds_same_train_urls_each_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    sources: list[_FakeHFSource] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> _FakeHFSource:
        _ = args
        calls.append(dict(kwargs))
        source = _FakeHFSource([_sample_row()])
        sources.append(source)
        return source

    def scene_splits(_: str, __: int) -> dict[str, tuple[SceneShard, ...]]:
        return {
            "train": (
                SceneShard("spring", "1"),
                SceneShard("summer", "2"),
                SceneShard("fall", "3"),
            ),
            "val": (SceneShard("winter", "4"),),
            "test": (SceneShard("winter", "5"),),
        }

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(data_mod, "get_token", lambda: "hf_test_token")

    train_loader, _, _ = build_sen12mscr_loaders(
        1,
        seed=13,
        num_workers=0,
        _scene_split_resolver=scene_splits,
    )

    train_loader.dataset.set_epoch(0)
    _ = next(iter(train_loader))
    train_loader.dataset.set_epoch(1)
    _ = next(iter(train_loader))

    assert len(calls) == 2
    assert calls[0]["data_files"] == calls[1]["data_files"]
    assert calls[0]["data_files"]["train"] == [
        SceneShard("spring", "1").resolve_url(),
        SceneShard("summer", "2").resolve_url(),
        SceneShard("fall", "3").resolve_url(),
    ]
    assert sources[0].shuffle_calls == [(13, DEFAULT_SHUFFLE_BUFFER_SIZE)]
    assert sources[1].shuffle_calls == [(13, DEFAULT_SHUFFLE_BUFFER_SIZE)]
    assert sources[0].epochs == [0]
    assert sources[1].epochs == [1]


def test_default_hf_map_loader_uses_official_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_dataset(*args: object, **kwargs: object) -> list[dict[str, object]]:
        _ = args
        calls.append(dict(kwargs))
        scene = kwargs["data_files"]["train"][0].rsplit("_", 1)[-1].split(".")[0]
        return [_sample_row(scene=scene)]

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(data_mod, "get_token", lambda: "hf_test_token")

    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        1,
        streaming=False,
        num_workers=0,
        _scene_split_resolver=_fake_scene_splits,
    )

    batch = next(iter(train_loader))

    assert isinstance(train_loader.dataset, list)
    assert isinstance(val_loader.dataset, list)
    assert isinstance(test_loader.dataset, list)
    assert len(calls) == 3
    for call in calls:
        assert call["streaming"] is False
        assert call["split"] == "train"
        assert call["columns"] == list(PARQUET_COLUMNS)
        assert call["token"] == "hf_test_token"
        assert "batch_size" not in call
    assert tuple(batch["target"].shape) == (1, 13, 2, 1)


def test_local_parquet_smoke_streaming_and_map_loading(tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    row = _sample_row()
    table = pa.table(
        {
            "sar": [row["sar"]],
            "cloudy": [row["cloudy"]],
            "target": [row["target"]],
            "sar_shape": [row["sar_shape"]],
            "opt_shape": [row["opt_shape"]],
            "season": [row["season"]],
            "scene": [row["scene"]],
            "patch": [row["patch"]],
        }
    )
    pq.write_table(table, parquet_path)

    streaming_source = data_mod._load_parquet_source(
        [str(parquet_path)],
        streaming=True,
    )
    mapped_source = data_mod._load_parquet_source(
        [str(parquet_path)],
        streaming=False,
    )

    assert [item["patch"] for item in streaming_source] == ["p30"]
    assert len(mapped_source) == 1
    assert mapped_source[0]["patch"] == "p30"


def test_minimal_train_cli_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "minimal_train.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "--num-workers" in result.stdout
    assert "--io-profile" not in result.stdout
