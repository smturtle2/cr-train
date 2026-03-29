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
from cr_train import build_sen12mscr_loaders
from cr_train.data import (
    DEFAULT_DATASET_REVISION,
    SEN12MSCRStreamingDataset,
    SceneShard,
    decode_sample,
    official_scene_splits,
    seeded_scene_splits,
)


def _sample_row(*, season: str = "spring", scene: str = "1", patch: str = "p30") -> dict[str, object]:
    sar = np.array(
        [
            [[-30.0, -12.5], [-25.0, 0.0]],
            [[-5.0, 1.0], [-7.5, -20.0]],
        ],
        dtype=np.float32,
    )
    # CHW: 13 channels, 1x2 spatial
    optical = np.arange(13 * 1 * 2, dtype=np.int16).reshape(13, 1, 2) * 800 - 10
    return {
        "sar": sar.tobytes(),
        "cloudy": optical.tobytes(),
        "target": (optical + 100).tobytes(),
        "sar_shape": list(sar.shape),
        "opt_shape": list(optical.shape),
        "dtype": "float32",
        "season": season,
        "scene": scene,
        "patch": patch,
    }


def test_decode_sample_applies_official_preprocessing_and_standard_schema() -> None:
    decoded = decode_sample(_sample_row())

    assert decoded["inputs"]["sar"].shape == (2, 2, 2)
    assert decoded["inputs"]["cloudy"].shape == (13, 1, 2)
    assert decoded["target"].shape == (13, 1, 2)
    assert decoded["metadata"]["source_shard"] == "spring/scene_1.parquet"
    assert decoded["metadata"]["patch"] == "p30"
    assert decoded["inputs"]["sar"].dtype == torch.float32
    assert decoded["inputs"]["cloudy"].dtype == torch.float32
    assert decoded["target"].dtype == torch.float32
    assert torch.isclose(decoded["inputs"]["sar"].min(), torch.tensor(0.0))
    assert torch.isclose(decoded["inputs"]["sar"].max(), torch.tensor(1.0))
    assert torch.isclose(decoded["inputs"]["cloudy"].min(), torch.tensor(0.0))
    assert torch.isclose(decoded["inputs"]["cloudy"].max(), torch.tensor(1.0))
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


def test_official_scene_splits_return_a_fresh_mapping() -> None:
    splits = official_scene_splits()
    splits["train"] = ()

    fresh = official_scene_splits()

    assert len(fresh["train"]) == 155


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


class _FakeIterable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[object, ...]] = []

    def reshard(self) -> "_FakeIterable":
        self.calls.append(("reshard",))
        return self

    def shuffle(self, *, seed: int, buffer_size: int) -> "_FakeIterable":
        self.calls.append(("shuffle", seed, buffer_size))
        return self

    def set_epoch(self, epoch: int) -> None:
        self.calls.append(("set_epoch", epoch))

    def __iter__(self):
        return iter(self.rows)


class _EpochAwareSource:
    def __init__(self) -> None:
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        yield _sample_row(scene=str(self.epoch), patch=f"p{self.epoch}")


def test_build_loaders_defaults_to_official_and_reshards_before_shuffle() -> None:
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
        return {
            "train": (SceneShard("spring", "1"),),
            "val": (SceneShard("spring", "2"),),
            "test": (SceneShard("spring", "3"),),
        }

    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        1,
        seed=13,
        num_workers=0,
        shuffle_buffer_size=16,
        _dataset_loader=fake_dataset_loader,
        _scene_split_resolver=fake_scene_split_resolver,
    )

    train_loader.dataset.set_epoch(3)
    batch = next(iter(train_loader))
    _ = next(iter(val_loader))
    _ = next(iter(test_loader))

    assert resolver_calls == [("official", 13)]
    assert tuple(batch["inputs"]["sar"].shape) == (1, 2, 2, 2)
    assert tuple(batch["target"].shape) == (1, 13, 1, 2)
    assert datasets["train"].calls[:3] == [
        ("reshard",),
        ("shuffle", 13, 16),
        ("set_epoch", 3),
    ]
    assert datasets["val"].calls == [("set_epoch", 0)]
    assert datasets["test"].calls == [("set_epoch", 0)]


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
    assert datasets["train"].calls[:2] == [("reshard",), ("shuffle", 7, 16)]
    assert datasets["val"].calls == [("set_epoch", 0)]
    assert datasets["test"].calls == [("set_epoch", 0)]


def test_loader_builder_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(0)
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, split="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, num_workers=-1)
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, num_workers=2, prefetch_factor=0)
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, num_workers=0, prefetch_factor=2)
    with pytest.raises(ValueError):
        build_sen12mscr_loaders(1, num_workers=0, persistent_workers=True)


def test_loader_builder_auto_tunes_worker_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
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
            self.kwargs = kwargs

        def __class_getitem__(cls, _: object) -> type["_FakeDataLoader"]:
            return cls

    def fake_dataset_loader(urls: list[str], stage: str) -> _FakeIterable:
        _ = urls
        return datasets[stage]

    def fake_scene_split_resolver(split: str, seed: int) -> dict[str, tuple[SceneShard, ...]]:
        _ = (split, seed)
        return {
            "train": (SceneShard("spring", "1"), SceneShard("spring", "2")),
            "val": (SceneShard("spring", "3"),),
            "test": (SceneShard("spring", "4"), SceneShard("spring", "5")),
        }

    monkeypatch.setattr(data_mod, "DataLoader", _FakeDataLoader)
    monkeypatch.setattr(data_mod.os, "cpu_count", lambda: 12)

    build_sen12mscr_loaders(
        2,
        _dataset_loader=fake_dataset_loader,
        _scene_split_resolver=fake_scene_split_resolver,
    )

    assert [kwargs["num_workers"] for kwargs in created] == [2, 0, 0]
    for index, kwargs in enumerate(created):
        assert kwargs["batch_size"] == 2
        assert kwargs["worker_init_fn"] is data_mod._seed_worker
        assert isinstance(kwargs["generator"], torch.Generator)
        if kwargs["num_workers"] > 0:
            assert kwargs["prefetch_factor"] == 2
            assert kwargs["persistent_workers"] is True
        else:
            assert "prefetch_factor" not in kwargs
            assert "persistent_workers" not in kwargs


def test_loader_builder_passes_prefetch_and_persistent_worker_options(
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
            self.kwargs = kwargs

        def __class_getitem__(cls, _: object) -> type["_FakeDataLoader"]:
            return cls

    def fake_dataset_loader(urls: list[str], stage: str) -> _FakeIterable:
        _ = urls
        return datasets[stage]

    def fake_scene_split_resolver(split: str, seed: int) -> dict[str, tuple[SceneShard, ...]]:
        _ = (split, seed)
        return {
            "train": (SceneShard("spring", "1"),),
            "val": (SceneShard("spring", "2"),),
            "test": (SceneShard("spring", "3"),),
        }

    monkeypatch.setattr(data_mod, "DataLoader", _FakeDataLoader)

    build_sen12mscr_loaders(
        2,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        _dataset_loader=fake_dataset_loader,
        _scene_split_resolver=fake_scene_split_resolver,
    )

    assert len(created) == 3
    for kwargs in created:
        assert kwargs["batch_size"] == 2
        assert kwargs["num_workers"] == 2
        assert kwargs["pin_memory"] is True
        assert kwargs["worker_init_fn"] is data_mod._seed_worker
        assert isinstance(kwargs["generator"], torch.Generator)
        assert kwargs["prefetch_factor"] == 4
        assert kwargs["persistent_workers"] is True


@pytest.mark.filterwarnings("ignore:This process .* multi-threaded, use of fork")
def test_streaming_dataset_shares_epoch_updates_with_persistent_workers() -> None:
    dataset = SEN12MSCRStreamingDataset(_EpochAwareSource())
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        persistent_workers=True,
    )

    try:
        observed_scenes: list[str] = []
        for epoch in range(3):
            dataset.set_epoch(epoch)
            sample = next(iter(loader))
            observed_scenes.append(sample["metadata"]["scene"])
        assert observed_scenes == ["0", "1", "2"]
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


def test_seed_worker_keeps_numpy_seed_in_uint32_range(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(data_mod.torch, "initial_seed", lambda: (2**32) - 1)
    monkeypatch.setattr(data_mod.random, "seed", lambda value: calls.append(("python", value)))
    monkeypatch.setattr(data_mod.np.random, "seed", lambda value: calls.append(("numpy", value)))

    data_mod._seed_worker(1)

    assert calls == [("python", (2**32) - 1), ("numpy", (2**32) - 1)]


def test_runtime_is_not_configured_on_import() -> None:
    runtime_mod = importlib.import_module("cr_train.runtime")
    runtime_mod = importlib.reload(runtime_mod)

    assert runtime_mod._CONFIGURED is False


def test_runtime_configuration_rejects_conflicting_io_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_mod = importlib.import_module("cr_train.runtime")
    runtime_mod = importlib.reload(runtime_mod)
    patched_profiles: list[str] = []

    monkeypatch.setattr(
        runtime_mod,
        "_patch_datasets_parquet_reader",
        lambda io_profile: patched_profiles.append(io_profile),
    )

    runtime_mod.configure_runtime("smooth")
    runtime_mod.configure_runtime("smooth")

    with pytest.raises(ValueError, match="runtime already configured"):
        runtime_mod.configure_runtime("conservative")

    assert patched_profiles == ["smooth"]


def test_runtime_smooth_profile_uses_notebook_safe_scan_behavior() -> None:
    runtime_mod = importlib.import_module("cr_train.runtime")
    runtime_mod = importlib.reload(runtime_mod)

    assert runtime_mod._scan_behavior("smooth") == (1, 1, False)
    assert runtime_mod._scan_behavior("conservative") == (0, 0, False)


def test_default_dataset_loader_bootstraps_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    dummy_iterable = _FakeIterable([])

    def fake_configure_runtime(*, io_profile: str = "smooth") -> None:
        calls.append(("configure", io_profile))

    def fake_load_dataset(*args: object, **kwargs: object) -> _FakeIterable:
        calls.append(("load_dataset", kwargs["data_files"], kwargs["split"], kwargs["streaming"]))
        return dummy_iterable

    monkeypatch.setattr(data_mod, "configure_runtime", fake_configure_runtime)
    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)

    dataset = data_mod._default_dataset_loader(
        ["/tmp/sample.parquet"],
        "train",
        io_profile="conservative",
    )

    assert dataset is dummy_iterable
    assert calls == [
        ("configure", "conservative"),
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
