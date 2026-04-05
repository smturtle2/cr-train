from __future__ import annotations

import hashlib
import importlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from datasets import load_dataset

from cr_train.data import (
    BLOCK_SIZE,
    DATA_COLUMNS,
    PreparedSplit,
    build_collate_fn,
    build_dataloader,
    decode_row,
    plan_sample,
    trace_plan_sample,
)
from cr_train.data.dataset import prepare_split, prepare_split_from_state, resolve_prepared_split_state
from cr_train.data.runtime import ensure_split_cache
from cr_train.data.store import (
    block_data_path,
    block_is_cached,
    block_metadata_path,
    load_block,
    load_block_metadata,
    resolve_block_cache_paths,
    save_block,
)


def _make_row(index: int) -> dict[str, object]:
    sar = (np.arange(256 * 256 * 2, dtype=np.float32) + index).reshape(256, 256, 2)
    cloudy = (np.arange(256 * 256 * 13, dtype=np.int16) + index).reshape(256, 256, 13)
    target = (np.arange(256 * 256 * 13, dtype=np.int16) + index + 10).reshape(256, 256, 13)
    return {
        "sar": sar.tobytes(),
        "cloudy": cloudy.tobytes(),
        "target": target.tobytes(),
        "sar_shape": [256, 256, 2],
        "opt_shape": [256, 256, 13],
        "season": "spring",
        "scene": str(index),
        "patch": f"p{index:03d}",
    }


def _make_stream_row(index: int) -> dict[str, object]:
    sar = bytes(((index + offset) % 251 for offset in range(2)))
    cloudy = bytes(((index + offset) % 251 for offset in range(13)))
    target = bytes(((index + offset + 17) % 251 for offset in range(13)))
    return {
        "sar": sar,
        "cloudy": cloudy,
        "target": target,
        "sar_shape": [1, 1, 2],
        "opt_shape": [1, 1, 13],
        "season": "spring",
        "scene": str(index),
        "patch": f"p{index:03d}",
    }


class _RowDataset(torch.utils.data.Dataset[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = tuple(dict(row) for row in rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return dict(self.rows[index])


def _load_local_parquet_stream(path: Path):
    return load_dataset(
        "parquet",
        data_files=str(path),
        split="train",
        streaming=True,
        columns=DATA_COLUMNS,
    )


def _count_generated_rows(generate_tables_fn, path: Path, row_groups: tuple[int, ...]) -> int:
    return sum(table.num_rows for _, table in generate_tables_fn([str(path)], [row_groups]))


@pytest.fixture
def parquet_row_group_path(tmp_path: Path) -> Path:
    path = tmp_path / "scene_77.parquet"
    rows = [_make_stream_row(index) for index in range((11 * BLOCK_SIZE) + 11)]
    table = pa.table({column: [row[column] for row in rows] for column in rows[0]})
    pq.write_table(table, path, row_group_size=BLOCK_SIZE)
    return path


def _selection_seed(seed: int, *, split: str) -> int:
    digest = hashlib.sha256(f"selection:{split}".encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "big")


def _uniform_selected_blocks(seed: int, *, split: str, required_blocks: int, total_blocks: int) -> np.ndarray:
    if required_blocks <= 0 or total_blocks <= 0:
        return np.empty((0,), dtype=np.int64)
    if required_blocks >= total_blocks:
        return np.arange(total_blocks, dtype=np.int64)
    rng = np.random.default_rng(_selection_seed(seed, split=split))
    return np.sort(rng.choice(total_blocks, size=required_blocks, replace=False).astype(np.int64))


def _make_block_splits(block_count: int) -> list[list[dict[str, object]]]:
    blocks: list[list[dict[str, object]]] = []
    current_index = 0
    for _ in range(block_count):
        block_rows = [_make_row(current_index + offset) for offset in range(BLOCK_SIZE)]
        blocks.append(block_rows)
        current_index += BLOCK_SIZE
    return blocks


def _catalog(split: str, blocks: list[list[dict[str, object]]]) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
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
        assert dataset_name == "unit/test" or dataset_name == "Hermanni/sen12mscr"
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
        "descriptor": descriptor,
        "catalogs": catalogs,
        "rows_by_key": rows_by_key,
        "load_counts": load_counts,
    }


def test_data_package_public_surface_is_minimal_and_explicit() -> None:
    data_mod = importlib.import_module("cr_train.data")

    assert set(data_mod.__all__) == {
        "BLOCK_SIZE",
        "CachedRowDataset",
        "DATASET_ID",
        "DATA_COLUMNS",
        "OPTICAL_CHANNELS",
        "PreparedSplit",
        "SAR_CHANNELS",
        "SamplePlan",
        "SelectionTrace",
        "build_collate_fn",
        "build_dataloader",
        "decode_row",
        "move_batch_to_device",
        "plan_sample",
        "resolve_num_workers",
        "seed_everything",
        "seed_worker",
        "trace_plan_sample",
    }
    assert all(hasattr(data_mod, name) for name in data_mod.__all__)


def test_decode_row_converts_to_chw_and_normalizes() -> None:
    decoded = decode_row(_make_row(3))

    assert decoded["sar"].shape == (2, 256, 256)
    assert decoded["cloudy"].shape == (13, 256, 256)
    assert decoded["target"].shape == (13, 256, 256)
    assert decoded["sar"].dtype == np.float32
    assert decoded["cloudy"].dtype == np.float32
    assert decoded["target"].dtype == np.float32
    assert decoded["meta"]["scene"] == "3"


def test_build_collate_fn_batches_rows() -> None:
    collate = build_collate_fn()
    batch = collate([_make_row(0), _make_row(1)])

    assert batch["sar"].shape == (2, 2, 256, 256)
    assert batch["cloudy"].shape == (2, 13, 256, 256)
    assert batch["target"].shape == (2, 13, 256, 256)
    assert batch["meta"]["patch"] == ["p000", "p001"]


def test_build_collate_fn_is_picklable() -> None:
    collate = build_collate_fn(crop_size=128, crop_mode="center")

    restored = pickle.loads(pickle.dumps(collate))
    batch = restored([_make_row(0)])

    assert batch["cloudy"].shape == (1, 13, 128, 128)


def test_build_collate_fn_applies_spatial_transforms_consistently(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    randint_values = iter((5, 7))
    random_values = iter((0.4, 0.6))
    monkeypatch.setattr(dataset_mod.random, "randint", lambda _lo, _hi: next(randint_values))
    monkeypatch.setattr(dataset_mod.random, "random", lambda: next(random_values))
    monkeypatch.setattr(dataset_mod.random, "randrange", lambda _stop: 1)

    row = _make_row(0)
    collate = build_collate_fn(
        crop_size=128,
        crop_mode="random",
        random_flip=True,
        random_rot90=True,
    )
    batch = collate([row])
    decoded = decode_row(row)

    expected_sar = torch.from_numpy(decoded["sar"][:, 5:133, 7:135].copy())
    expected_sar = torch.flip(expected_sar, dims=(-2,))
    expected_sar = torch.rot90(expected_sar, k=1, dims=(-2, -1))

    expected_cloudy = torch.from_numpy(decoded["cloudy"][:, 5:133, 7:135].copy())
    expected_cloudy = torch.flip(expected_cloudy, dims=(-2,))
    expected_cloudy = torch.rot90(expected_cloudy, k=1, dims=(-2, -1))

    expected_target = torch.from_numpy(decoded["target"][:, 5:133, 7:135].copy())
    expected_target = torch.flip(expected_target, dims=(-2,))
    expected_target = torch.rot90(expected_target, k=1, dims=(-2, -1))

    assert batch["sar"].shape == (1, 2, 128, 128)
    assert batch["cloudy"].shape == (1, 13, 128, 128)
    assert batch["target"].shape == (1, 13, 128, 128)
    torch.testing.assert_close(batch["sar"][0], expected_sar)
    torch.testing.assert_close(batch["cloudy"][0], expected_cloudy)
    torch.testing.assert_close(batch["target"][0], expected_target)


def test_build_collate_fn_center_crop_reduces_spatial_size() -> None:
    row = _make_row(1)
    collate = build_collate_fn(crop_size=128, crop_mode="center")
    batch = collate([row])
    decoded = decode_row(row)

    expected = torch.from_numpy(decoded["cloudy"][:, 64:192, 64:192].copy())

    assert batch["cloudy"].shape == (1, 13, 128, 128)
    torch.testing.assert_close(batch["cloudy"][0], expected)


def test_build_collate_fn_rejects_oversized_crop() -> None:
    collate = build_collate_fn(crop_size=512, crop_mode="random")

    with pytest.raises(ValueError, match="crop_size"):
        collate([_make_row(0)])


def test_build_collate_fn_requires_crop_size_for_random_or_center_crop() -> None:
    with pytest.raises(ValueError, match="crop_size"):
        build_collate_fn(crop_mode="random")

    with pytest.raises(ValueError, match="crop_size"):
        build_collate_fn(crop_mode="center")


def test_block_payload_round_trip_uses_mmap_layout(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    cache_paths = resolve_block_cache_paths(source_root, "train")
    rows = [_make_row(0), _make_row(1)]

    save_result = save_block(
        cache_paths,
        cache_key="block-key",
        rows=rows,
        metadata={
            "cache_key": "block-key",
            "split": "train",
            "block_index": 0,
            "shard_index": 0,
            "source_file": "hf://datasets/unit/test/train/0000.parquet",
            "row_groups": [0],
            "row_count": len(rows),
        },
    )

    payload_path = block_data_path(cache_paths, "block-key")
    payload = load_block(cache_paths, "block-key")
    first_row = payload[0]

    assert payload_path.is_dir()
    assert sorted(path.name for path in payload_path.iterdir()) == ["cloudy.npy", "payload.json", "sar.npy", "target.npy"]
    assert save_result.payload_bytes > 0
    assert len(payload) == 2
    assert isinstance(first_row["sar"], np.ndarray)
    np.testing.assert_array_equal(
        first_row["sar"],
        np.frombuffer(rows[0]["sar"], dtype=np.float32).reshape(rows[0]["sar_shape"]),
    )
    np.testing.assert_array_equal(
        first_row["cloudy"],
        np.frombuffer(rows[0]["cloudy"], dtype=np.int16).reshape(rows[0]["opt_shape"]),
    )
    np.testing.assert_array_equal(
        first_row["target"],
        np.frombuffer(rows[0]["target"], dtype=np.int16).reshape(rows[0]["opt_shape"]),
    )
    assert first_row["scene"] == "0"
    assert first_row["patch"] == "p000"


def test_block_is_cached_requires_complete_payload_files(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    cache_paths = resolve_block_cache_paths(source_root, "train")
    rows = [_make_row(0)]

    save_block(
        cache_paths,
        cache_key="block-key",
        rows=rows,
        metadata={
            "cache_key": "block-key",
            "split": "train",
            "block_index": 0,
            "shard_index": 0,
            "source_file": "hf://datasets/unit/test/train/0000.parquet",
            "row_groups": [0],
            "row_count": len(rows),
        },
    )

    payload_path = block_data_path(cache_paths, "block-key")

    assert block_is_cached(cache_paths, "block-key") is True
    (payload_path / "sar.npy").unlink()
    assert block_is_cached(cache_paths, "block-key") is False


def test_load_block_does_not_call_torch_load(monkeypatch, tmp_path: Path) -> None:
    import torch

    source_root = tmp_path / "source"
    cache_paths = resolve_block_cache_paths(source_root, "train")
    rows = [_make_row(0)]

    save_block(
        cache_paths,
        cache_key="block-key",
        rows=rows,
        metadata={
            "cache_key": "block-key",
            "split": "train",
            "block_index": 0,
            "shard_index": 0,
            "source_file": "hf://datasets/unit/test/train/0000.parquet",
            "row_groups": [0],
            "row_count": len(rows),
        },
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("torch.load should not run")))

    payload = load_block(cache_paths, "block-key")

    assert len(payload) == 1
    assert payload[0]["scene"] == "0"


def test_plan_sample_is_block_reproducible_within_total_block_domain() -> None:
    catalog = {"total_rows": 10 * BLOCK_SIZE, "total_blocks": 10}

    requested_rows = (2 * BLOCK_SIZE) - 1
    sample_a = plan_sample(catalog, seed=7, max_samples=requested_rows)
    sample_b = plan_sample(catalog, seed=7, max_samples=requested_rows)
    distinct_plans = {
        tuple(plan_sample(catalog, seed=seed, max_samples=3 * BLOCK_SIZE).selected_blocks.tolist())
        for seed in range(64)
    }

    assert sample_a.requested_rows == requested_rows
    assert sample_a.required_blocks == 2
    assert sample_a.effective_rows == 2 * BLOCK_SIZE
    assert sample_a.total_blocks == 10
    assert sample_a.planner_mode == "uniform_exact_k"
    assert sample_a.selected_blocks.size == sample_a.required_blocks
    assert np.array_equal(sample_a.selected_blocks, sample_b.selected_blocks)
    assert len(distinct_plans) > 1
    assert np.all(sample_a.selected_blocks < sample_a.total_blocks)
    assert sample_a.execution_block_count == int(sample_a.selected_blocks[-1]) + 1


def test_trace_plan_sample_reports_uniform_exact_k_metadata() -> None:
    trace = trace_plan_sample({"total_rows": 10 * BLOCK_SIZE}, seed=11, max_samples=3 * BLOCK_SIZE, split="train")

    assert trace.total_blocks == 10
    assert trace.requested_rows == 3 * BLOCK_SIZE
    assert trace.required_blocks == 3
    assert trace.planner_mode == "uniform_exact_k"
    assert trace.selected_blocks.size == 3
    assert trace.draw_order.size == 3
    assert trace.execution_block_count == int(trace.selected_blocks[-1]) + 1


def test_prepare_row_group_stream_activates_internal_row_group_reshard(monkeypatch) -> None:
    import cr_train.data.source as source_mod

    patch_calls = []

    class FakeExIterable:
        def __init__(self, kwargs: dict[str, object]) -> None:
            self.kwargs = kwargs
            self.generate_tables_fn = object()
            self.generate_more_kwargs_fn = object()

    class FakeDataset:
        def __init__(self, *, num_shards: int, kwargs: dict[str, object]) -> None:
            self.num_shards = num_shards
            self._ex_iterable = FakeExIterable(kwargs)
            self._info = SimpleNamespace(copy=lambda: self._info)
            self._split = "train"

        def reshard(self):
            row_groups_list = self._ex_iterable.kwargs["row_groups_list"]
            if row_groups_list is None:
                return FakeDataset(
                    num_shards=13,
                    kwargs={
                        "files": ["file.parquet"] * 13,
                        "row_groups_list": [(index,) for index in range(13)],
                    },
                )
            return FakeDataset(num_shards=self.num_shards, kwargs=dict(self._ex_iterable.kwargs))

    monkeypatch.setattr(
        source_mod,
        "_clone_iterable_dataset",
        lambda dataset, kwargs: FakeDataset(num_shards=dataset.num_shards, kwargs=kwargs),
    )
    monkeypatch.setattr(source_mod, "_patch_parquet_generate_tables_fn", lambda dataset: patch_calls.append(dataset))

    dataset = FakeDataset(num_shards=1, kwargs={"files": ["file.parquet"], "row_groups_list": [None]})
    resharded = source_mod._prepare_row_group_stream(dataset)

    assert patch_calls == [dataset]
    assert resharded.num_shards == 13
    assert resharded._ex_iterable.kwargs["row_groups_list"][0] == (0,)


def test_supported_datasets_version_is_enforced(monkeypatch) -> None:
    import cr_train.data.source as source_mod

    monkeypatch.setattr(source_mod, "_installed_datasets_version", lambda: "4.8.0")

    with pytest.raises(RuntimeError, match="datasets==4.7.0"):
        source_mod.ensure_supported_datasets_version()


def test_plain_parquet_generate_tables_ignores_row_group_subset(parquet_row_group_path: Path) -> None:
    stream = _load_local_parquet_stream(parquet_row_group_path)

    row_count = _count_generated_rows(stream._ex_iterable.generate_tables_fn, parquet_row_group_path, (4,))

    assert row_count == (11 * BLOCK_SIZE) + 11


def test_patched_parquet_generate_tables_respects_row_group_subset(parquet_row_group_path: Path) -> None:
    import cr_train.data.source as source_mod

    stream = _load_local_parquet_stream(parquet_row_group_path)
    source_mod._patch_parquet_generate_tables_fn(stream)
    source_mod._patch_parquet_generate_tables_fn(stream)

    assert _count_generated_rows(stream._ex_iterable.generate_tables_fn, parquet_row_group_path, (4,)) == BLOCK_SIZE
    assert _count_generated_rows(stream._ex_iterable.generate_tables_fn, parquet_row_group_path, (11,)) == 11


def test_prepare_row_group_stream_fetches_one_row_group_per_shard(parquet_row_group_path: Path) -> None:
    import cr_train.data.source as source_mod

    stream = source_mod._prepare_row_group_stream(_load_local_parquet_stream(parquet_row_group_path))

    assert int(stream.num_shards) == 12
    assert len(list(stream.shard(num_shards=int(stream.num_shards), index=4, contiguous=True).take(BLOCK_SIZE + 1))) == BLOCK_SIZE
    assert len(list(stream.shard(num_shards=int(stream.num_shards), index=11, contiguous=True).take(BLOCK_SIZE + 1))) == 11


def test_load_block_rows_uses_shard_index(monkeypatch, parquet_row_group_path: Path) -> None:
    import cr_train.data.source as source_mod

    stream = source_mod._prepare_row_group_stream(_load_local_parquet_stream(parquet_row_group_path))
    monkeypatch.setattr(source_mod, "load_row_group_stream", lambda **kwargs: stream)

    rows = source_mod.load_block_rows(
        dataset_name="unit/test",
        revision=None,
        split="train",
        block={
            "index": 0,
            "shard_index": 11,
            "cache_key": "tail-block",
            "source_file": str(parquet_row_group_path),
            "row_groups": [11],
        },
    )

    assert len(rows) == 11
    assert rows[0]["scene"] == str(11 * BLOCK_SIZE)


def test_load_block_rows_rejects_empty_shard(monkeypatch) -> None:
    import cr_train.data.source as source_mod

    class FakeShardDataset:
        def __init__(self, rows):
            self.rows = rows

        def take(self, count: int):
            return list(self.rows[:count])

    class FakeTemplate:
        num_shards = 8

        def shard(self, *, num_shards: int, index: int, contiguous: bool):
            assert num_shards == 8
            assert contiguous is True
            assert index == 7
            return FakeShardDataset([])

    monkeypatch.setattr(source_mod, "load_row_group_stream", lambda **kwargs: FakeTemplate())

    with pytest.raises(RuntimeError) as exc_info:
        source_mod.load_block_rows(
            dataset_name="unit/test",
            revision=None,
            split="train",
            block={
                "index": 0,
                "shard_index": 7,
                "cache_key": "empty-block",
                "source_file": "hf://datasets/unit/test/train/0007.parquet",
                "row_groups": [7],
            },
        )

    message = str(exc_info.value)
    assert "cache_key=empty-block" in message
    assert "shard_index=7" in message
    assert "source_file=hf://datasets/unit/test/train/0007.parquet" in message
    assert "row_groups=[7]" in message


def test_load_block_rows_rejects_oversized_shard(monkeypatch) -> None:
    import cr_train.data.source as source_mod

    class FakeShardDataset:
        def __init__(self, rows):
            self.rows = rows

        def take(self, count: int):
            return list(self.rows[:count])

    class FakeTemplate:
        num_shards = 3

        def shard(self, *, num_shards: int, index: int, contiguous: bool):
            assert num_shards == 3
            assert contiguous is True
            assert index == 2
            return FakeShardDataset([_make_stream_row(row_index) for row_index in range(BLOCK_SIZE + 1)])

    monkeypatch.setattr(source_mod, "load_row_group_stream", lambda **kwargs: FakeTemplate())

    with pytest.raises(RuntimeError) as exc_info:
        source_mod.load_block_rows(
            dataset_name="unit/test",
            revision=None,
            split="train",
            block={
                "index": 1,
                "shard_index": 2,
                "cache_key": "oversized-block",
                "source_file": "hf://datasets/unit/test/train/0002.parquet",
                "row_groups": [2],
            },
        )

    message = str(exc_info.value)
    assert f"BLOCK_SIZE={BLOCK_SIZE}" in message
    assert "cache_key=oversized-block" in message
    assert "shard_index=2" in message
    assert "source_file=hf://datasets/unit/test/train/0002.parquet" in message
    assert "row_groups=[2]" in message


def test_build_catalog_records_shard_index(monkeypatch, parquet_row_group_path: Path) -> None:
    import cr_train.data.source as source_mod

    stream = source_mod._prepare_row_group_stream(_load_local_parquet_stream(parquet_row_group_path))
    monkeypatch.setattr(source_mod, "load_row_group_stream", lambda **kwargs: stream)

    catalog = source_mod.build_catalog(
        dataset_name="unit/test",
        revision=None,
        split="train",
        total_rows=(11 * BLOCK_SIZE) + 11,
    )

    assert len(catalog["blocks"]) == 12
    assert all("shard_index" in block for block in catalog["blocks"])
    assert all("cache_key" in block for block in catalog["blocks"])
    assert all("source_file" in block for block in catalog["blocks"])
    assert all("row_groups" in block for block in catalog["blocks"])
    assert catalog["blocks"][11]["shard_index"] == 11
    assert catalog["blocks"][11]["row_groups"] == [11]


def test_ensure_split_cache_reuses_cached_blocks_across_plans(monkeypatch, tmp_path: Path) -> None:
    split_blocks = {
        "train": _make_block_splits(4),
        "validation": _make_block_splits(2),
        "test": _make_block_splits(2),
    }
    patched = _patch_split_cache(monkeypatch, tmp_path, split_blocks)
    catalog = patched["catalogs"]["train"]
    first_seed = 7
    first_plan = plan_sample(catalog, seed=first_seed, max_samples=2 * BLOCK_SIZE, split="train")
    second_seed, second_plan = next(
        (seed, plan)
        for seed in range(8, 4096)
        for plan in [plan_sample(catalog, seed=seed, max_samples=2 * BLOCK_SIZE, split="train")]
        if 0 < len(set(first_plan.selected_blocks.tolist()) & set(plan.selected_blocks.tolist())) < len(plan.selected_blocks)
    )

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=first_seed,
        cache_root=tmp_path,
    )
    cache_paths = resolve_block_cache_paths(patched["source_root"], "train")
    cached_after_first = sorted(path.name for path in cache_paths.block_root.iterdir())

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=second_seed,
        cache_root=tmp_path,
    )
    cached_after_second = sorted(path.name for path in cache_paths.block_root.iterdir())

    assert len(cached_after_first) == len(first_plan.selected_blocks)
    assert len(cached_after_second) == len(set(first_plan.selected_blocks.tolist()) | set(second_plan.selected_blocks.tolist()))


@pytest.mark.parametrize(
    ("label", "mutate_metadata"),
    [
        (
            "row_count_overflow",
            lambda metadata: metadata.__setitem__("row_count", BLOCK_SIZE + 1),
        ),
        (
            "shard_index_mismatch",
            lambda metadata: metadata.__setitem__("shard_index", int(metadata["shard_index"]) + 1),
        ),
        (
            "source_file_row_groups_mismatch",
            lambda metadata: (
                metadata.__setitem__("source_file", "hf://datasets/unit/test/train/stale.parquet"),
                metadata.__setitem__("row_groups", [99]),
            ),
        ),
    ],
)
def test_ensure_split_cache_refills_stale_block_cache(
    monkeypatch,
    tmp_path: Path,
    label: str,
    mutate_metadata,
) -> None:
    del label
    split_blocks = {"train": _make_block_splits(1)}
    patched = _patch_split_cache(monkeypatch, tmp_path, split_blocks)
    block = patched["catalogs"]["train"]["blocks"][0]
    cache_key = str(block["cache_key"])

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=BLOCK_SIZE,
        seed=7,
        cache_root=tmp_path,
    )

    cache_paths = resolve_block_cache_paths(patched["source_root"], "train")
    metadata_path = block_metadata_path(cache_paths, cache_key)
    metadata = load_block_metadata(cache_paths, cache_key)

    assert metadata is not None
    assert patched["load_counts"][cache_key] == 1

    stale_metadata = dict(metadata)
    mutate_metadata(stale_metadata)
    metadata_path.write_text(json.dumps(stale_metadata, sort_keys=True, indent=2), encoding="utf-8")

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=BLOCK_SIZE,
        seed=7,
        cache_root=tmp_path,
    )

    refilled_metadata = load_block_metadata(cache_paths, cache_key)

    assert refilled_metadata is not None
    assert patched["load_counts"][cache_key] == 2
    assert refilled_metadata["shard_index"] == block["shard_index"]
    assert refilled_metadata["source_file"] == block["source_file"]
    assert refilled_metadata["row_groups"] == block["row_groups"]
    assert 0 < int(refilled_metadata["row_count"]) <= BLOCK_SIZE


def test_ensure_split_cache_refills_missing_payload_file(monkeypatch, tmp_path: Path) -> None:
    split_blocks = {"train": _make_block_splits(1)}
    patched = _patch_split_cache(monkeypatch, tmp_path, split_blocks)
    block = patched["catalogs"]["train"]["blocks"][0]
    cache_key = str(block["cache_key"])

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=BLOCK_SIZE,
        seed=7,
        cache_root=tmp_path,
    )

    cache_paths = resolve_block_cache_paths(patched["source_root"], "train")
    payload_path = block_data_path(cache_paths, cache_key)
    (payload_path / "sar.npy").unlink()

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=BLOCK_SIZE,
        seed=7,
        cache_root=tmp_path,
    )

    assert patched["load_counts"][cache_key] == 2
    assert block_is_cached(cache_paths, cache_key) is True


def test_prepare_split_reads_cached_blocks_in_selected_order(monkeypatch, tmp_path: Path) -> None:
    split_blocks = {
        "train": _make_block_splits(4),
        "validation": _make_block_splits(4),
        "test": _make_block_splits(4),
    }
    patched = _patch_split_cache(monkeypatch, tmp_path, split_blocks)

    ensure_split_cache(
        split="validation",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=13,
        cache_root=tmp_path,
    )
    prepared = prepare_split(
        split="validation",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=13,
        epoch=0,
        training=False,
        cache_root=tmp_path,
    )
    loader = build_dataloader(
        prepared,
        batch_size=8,
        num_workers=0,
        training=False,
        seed=13,
        epoch=0,
    )

    batch_scenes = [scene for batch in loader for scene in batch["meta"]["scene"]]
    sample_plan = plan_sample(
        patched["catalogs"]["validation"],
        seed=13,
        max_samples=2 * BLOCK_SIZE,
        split="validation",
    )
    expected_scenes: list[str] = []
    for block_index in sample_plan.selected_blocks.tolist():
        block = patched["catalogs"]["validation"]["blocks"][int(block_index)]
        expected_scenes.extend(row["scene"] for row in patched["rows_by_key"][str(block["cache_key"])])

    assert batch_scenes == expected_scenes


def test_prepare_split_training_order_changes_by_epoch(monkeypatch, tmp_path: Path) -> None:
    split_blocks = {
        "train": _make_block_splits(4),
        "validation": _make_block_splits(2),
        "test": _make_block_splits(2),
    }
    _patch_split_cache(monkeypatch, tmp_path, split_blocks)

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=9,
        cache_root=tmp_path,
    )
    state = resolve_prepared_split_state(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=9,
        cache_root=tmp_path,
    )
    prepared_epoch0 = prepare_split_from_state(
        state,
        epoch=0,
        training=True,
    )
    prepared_epoch1 = prepare_split_from_state(
        state,
        epoch=1,
        training=True,
    )
    loader_epoch0 = build_dataloader(
        prepared_epoch0,
        batch_size=8,
        num_workers=0,
        training=True,
        seed=9,
        epoch=0,
    )
    loader_epoch1 = build_dataloader(
        prepared_epoch1,
        batch_size=8,
        num_workers=0,
        training=True,
        seed=9,
        epoch=1,
    )

    scenes_epoch0 = [scene for batch in loader_epoch0 for scene in batch["meta"]["scene"]]
    scenes_epoch1 = [scene for batch in loader_epoch1 for scene in batch["meta"]["scene"]]

    assert scenes_epoch0 != scenes_epoch1
    assert set(scenes_epoch0) == set(scenes_epoch1)


def test_build_dataloader_defaults_to_non_persistent_workers(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    captured: dict[str, Any] = {}

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dataset_mod, "DataLoader", FakeDataLoader)

    prepared = PreparedSplit(dataset=SimpleNamespace(name="dataset"), num_examples=4)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=2,
        training=False,
        seed=5,
        epoch=0,
    )

    assert loader is not None
    assert captured["dataset"] is prepared.dataset
    assert captured["kwargs"]["persistent_workers"] is False
    assert "multiprocessing_context" not in captured["kwargs"]


def test_build_dataloader_passes_multiprocessing_context(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    captured: dict[str, Any] = {}

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dataset_mod, "DataLoader", FakeDataLoader)

    prepared = PreparedSplit(dataset=SimpleNamespace(name="dataset"), num_examples=4)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=2,
        training=False,
        seed=5,
        epoch=0,
        multiprocessing_context="spawn",
    )

    assert loader is not None
    assert captured["dataset"] is prepared.dataset
    assert captured["kwargs"]["multiprocessing_context"] == "spawn"


def test_build_dataloader_ignores_multiprocessing_context_without_workers(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    captured: dict[str, Any] = {}

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dataset_mod, "DataLoader", FakeDataLoader)

    prepared = PreparedSplit(dataset=SimpleNamespace(name="dataset"), num_examples=4)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=0,
        training=False,
        seed=5,
        epoch=0,
        multiprocessing_context="spawn",
    )

    assert loader is not None
    assert captured["dataset"] is prepared.dataset
    assert "multiprocessing_context" not in captured["kwargs"]


def test_build_dataloader_supports_spawn_workers() -> None:
    prepared = PreparedSplit(dataset=_RowDataset([_make_row(0), _make_row(1)]), num_examples=2)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=1,
        training=False,
        seed=5,
        epoch=0,
        multiprocessing_context="spawn",
    )

    batch = next(iter(loader))

    assert batch["sar"].shape == (2, 2, 256, 256)
    assert batch["cloudy"].shape == (2, 13, 256, 256)


def test_build_dataloader_passes_spatial_transform_options(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    captured: dict[str, Any] = {}

    def fake_build_collate_fn(**kwargs):
        captured["collate_kwargs"] = kwargs
        return "fake-collate"

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dataset_mod, "build_collate_fn", fake_build_collate_fn)
    monkeypatch.setattr(dataset_mod, "DataLoader", FakeDataLoader)

    prepared = PreparedSplit(dataset=SimpleNamespace(name="dataset"), num_examples=4)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=0,
        training=True,
        seed=5,
        epoch=0,
        crop_size=128,
        crop_mode="random",
        random_flip=True,
        random_rot90=True,
    )

    assert loader is not None
    assert captured["dataset"] is prepared.dataset
    assert captured["collate_kwargs"] == {
        "include_metadata": True,
        "crop_size": 128,
        "crop_mode": "random",
        "random_flip": True,
        "random_rot90": True,
    }
    assert captured["kwargs"]["collate_fn"] == "fake-collate"
