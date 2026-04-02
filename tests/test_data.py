from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from cr_train.data import (
    BLOCK_SIZE,
    STOP_BIAS_ALPHA,
    build_collate_fn,
    build_dataloader,
    compute_stop_probability,
    decode_row,
    ensure_source_root,
    ensure_split_cache,
    normalize_parquet_uri,
    plan_sample,
    prepare_split,
)
from cr_train.data.runtime import (
    WarmupProgressState,
    _compact_warmup_timeline,
    _render_warmup_timeline,
    _update_warmup_progress,
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


class _FakeTqdm:
    instances: list["_FakeTqdm"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")
        self.disable = kwargs.get("disable", False)
        self.updates: list[int] = []
        self.postfixes: list[str] = []
        self.desc = kwargs.get("desc")
        self.desc_history: list[str] = [str(self.desc)] if self.desc is not None else []
        _FakeTqdm.instances.append(self)

    @staticmethod
    def write(_message: str) -> None:
        return None

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
        rows: list[dict[str, object]],
        row_group_rows: list[int],
        stats: dict[str, object],
    ) -> None:
        self.url = url
        self._rows = [dict(row) for row in rows]
        self._row_group_rows = list(row_group_rows)
        self.metadata = _FakeParquetMetadata(row_group_rows)
        stats["opens"][url] += 1
        self._stats = stats

    def read_row_group(self, index: int, columns: list[str] | None = None):
        self._stats["read_row_groups"].append((self.url, int(index)))
        start = sum(self._row_group_rows[:index])
        stop = start + self._row_group_rows[index]
        rows = self._rows[start:stop]
        selected_columns = list(columns) if columns is not None else list(rows[0].keys())
        return pa.table({column: [row[column] for row in rows] for column in selected_columns})


def _patch_source(monkeypatch, split_rows: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    from cr_train.data.source import _source_descriptor_cache

    _source_descriptor_cache.clear()

    parquet_entries: list[dict[str, object]] = []
    rows_by_url: dict[str, list[dict[str, object]]] = {}
    row_groups_by_url: dict[str, list[int]] = {}
    stats: dict[str, object] = {
        "opens": defaultdict(int),
        "read_row_groups": [],
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
        rows_by_url[url] = [dict(row) for row in rows]
        row_groups_by_url[url] = [
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
        return _FakeParquetFile(
            str(url),
            rows=rows_by_url[str(url)],
            row_group_rows=row_groups_by_url[str(url)],
            stats=stats,
        )

    monkeypatch.setattr("cr_train.data.source.request_json", fake_request_json)
    monkeypatch.setattr("cr_train.data.source.pq.ParquetFile", fake_parquet_file)
    monkeypatch.setattr("cr_train.data.runtime.pq.ParquetFile", fake_parquet_file)
    return stats


def _catalog(total_rows: int) -> dict[str, object]:
    return {"total_rows": total_rows}


def _uniform_execution_block_count(seed: int, *, required_blocks: int, total_blocks: int) -> int:
    rng = np.random.default_rng(seed)
    selected_blocks = np.sort(
        rng.choice(total_blocks, size=required_blocks, replace=False).astype(np.int64)
    )
    return int(selected_blocks[-1]) + 1


def _row_store_root(catalog_path: Path, split: str) -> Path:
    source_root = catalog_path.parent.parent
    return source_root / "row_store" / split


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


def test_plan_sample_is_block_reproducible_within_total_block_domain() -> None:
    catalog = _catalog(total_rows=10 * BLOCK_SIZE)

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
    assert sample_a.planner_mode == "stop_biased_exact_k"
    assert sample_a.stop_bias_alpha == pytest.approx(STOP_BIAS_ALPHA)
    assert sample_a.selected_blocks.size == sample_a.required_blocks
    assert np.array_equal(sample_a.selected_blocks, sample_b.selected_blocks)
    assert np.array_equal(sample_a.selected_row_ids, sample_b.selected_row_ids)
    assert np.array_equal(sample_a.selected_row_offsets, sample_b.selected_row_offsets)
    assert len(distinct_plans) > 1
    assert np.all(sample_a.selected_blocks < sample_a.total_blocks)
    assert sample_a.execution_block_count == int(sample_a.selected_blocks[-1]) + 1


def test_plan_sample_uses_row_group_blocks() -> None:
    catalog = {
        "total_rows": (BLOCK_SIZE * 4) + (BLOCK_SIZE + 5) + (BLOCK_SIZE - 5) + BLOCK_SIZE,
        "shards": [
            {
                "global_start": 0,
                "global_stop": (BLOCK_SIZE * 4) + (BLOCK_SIZE + 5) + (BLOCK_SIZE - 5) + BLOCK_SIZE,
                "row_group_rows": [BLOCK_SIZE + 5, BLOCK_SIZE - 5, 2 * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
            }
        ],
    }

    sample = plan_sample(catalog, seed=11, max_samples=3 * BLOCK_SIZE, split="train")
    expected_blocks = []
    current = 0
    for size in catalog["shards"][0]["row_group_rows"]:
        expected_blocks.append(np.arange(current, current + size, dtype=np.int64))
        current += size

    for index in range(sample.required_blocks):
        start = int(sample.selected_row_offsets[index])
        stop = int(sample.selected_row_offsets[index + 1])
        block_rows = sample.selected_row_ids[start:stop]
        assert any(np.array_equal(block_rows, expected) for expected in expected_blocks)


def test_plan_sample_rounds_requested_rows_by_fixed_block_size() -> None:
    catalog = {
        "total_rows": 80 + 80 + 80,
        "shards": [
            {
                "global_start": 0,
                "global_stop": 80 + 80 + 80,
                "row_group_rows": [80, 80, 80],
            }
        ],
    }

    sample = plan_sample(catalog, seed=5, max_samples=65, split="train")

    assert sample.required_blocks == 2
    assert sample.total_blocks == 3


def test_compute_stop_probability_is_front_biased_and_normalized() -> None:
    stop_probs = [
        compute_stop_probability(4, 8, stop_block, stop_bias_alpha=STOP_BIAS_ALPHA)
        for stop_block in range(3, 8)
    ]

    assert sum(stop_probs) == pytest.approx(1.0)
    assert stop_probs == sorted(stop_probs, reverse=True)
    assert stop_probs[0] > stop_probs[-1]


def test_render_warmup_timeline_maps_selected_and_skipped_blocks() -> None:
    selected = np.asarray([1, 0, 1, 0, 1], dtype=np.bool_)

    assert _render_warmup_timeline(selected, stop_block=5) == "█░█░█"
    assert _render_warmup_timeline(selected, stop_block=3) == "█░█"
    assert _render_warmup_timeline(selected, stop_block=0) == ""


def test_compact_warmup_timeline_truncates_with_ellipsis() -> None:
    timeline = "█" * 20 + "░" * 20

    compact = _compact_warmup_timeline(timeline, max_chars=16)

    assert len(compact) == 16
    assert compact.startswith("█")
    assert compact.endswith("░")
    assert "…" in compact


def test_plan_sample_front_bias_favors_earlier_stop_blocks() -> None:
    stop_counts = np.zeros(9, dtype=np.int64)

    for seed in range(4096):
        sample_plan = plan_sample(_catalog(total_rows=10 * BLOCK_SIZE), seed=seed, max_samples=2 * BLOCK_SIZE)
        stop_counts[sample_plan.execution_block_count - 2] += 1

    assert stop_counts.tolist() == sorted(stop_counts.tolist(), reverse=True)
    assert int(stop_counts[0]) > int(stop_counts[-1])


def test_plan_sample_reduces_execution_span_vs_uniform_choice_baseline() -> None:
    catalog = _catalog(total_rows=64 * BLOCK_SIZE)
    sequential_counts = []
    uniform_counts = []

    for seed in range(256):
        sample_plan = plan_sample(catalog, seed=seed, max_samples=16 * BLOCK_SIZE)
        sequential_counts.append(sample_plan.execution_block_count)
        uniform_counts.append(
                _uniform_execution_block_count(
                    seed,
                    required_blocks=sample_plan.required_blocks,
                    total_blocks=sample_plan.total_blocks,
                )
            )

    assert float(np.mean(sequential_counts)) < float(np.mean(uniform_counts))


def test_normalize_parquet_uri_converts_hf_resolve_url() -> None:
    url = "https://huggingface.co/datasets/Hermanni/sen12mscr/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

    assert normalize_parquet_uri(url) == "hf://datasets/Hermanni/sen12mscr@refs/convert/parquet/default/train/0000.parquet"


def test_ensure_source_root_refreshes_when_remote_signature_changes(monkeypatch, tmp_path: Path) -> None:
    from cr_train.data.source import _source_descriptor_cache

    _source_descriptor_cache.clear()
    version = {"value": 0}

    def fake_request_json(url: str):
        if "/info?" in url:
            return {
                "dataset_info": {
                    "default": {
                        "splits": {
                            "train": {"num_examples": 64},
                        }
                    }
                }
            }
        if "/parquet?" in url:
            filename = "0000.parquet" if version["value"] == 0 else "0001.parquet"
            return {
                "parquet_files": [
                    {
                        "split": "train",
                        "url": f"hf://datasets/unit/test@refs/convert/parquet/default/train/{filename}",
                        "filename": filename,
                        "config": "default",
                    }
                ]
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("cr_train.data.source.request_json", fake_request_json)

    first_root, first_descriptor = ensure_source_root(
        dataset_name="unit/test",
        revision=None,
        cache_root=tmp_path,
    )

    version["value"] = 1
    _source_descriptor_cache.clear()
    second_root, second_descriptor = ensure_source_root(
        dataset_name="unit/test",
        revision=None,
        cache_root=tmp_path,
    )

    assert first_descriptor["source_signature"] != second_descriptor["source_signature"]
    assert first_root != second_root


def test_ensure_source_root_falls_back_to_cached_descriptor_on_refresh_failure(monkeypatch, tmp_path: Path) -> None:
    from cr_train.data.source import _source_descriptor_cache

    _source_descriptor_cache.clear()

    def initial_request_json(url: str):
        if "/info?" in url:
            return {
                "dataset_info": {
                    "default": {
                        "splits": {
                            "train": {"num_examples": 64},
                        }
                    }
                }
            }
        if "/parquet?" in url:
            return {
                "parquet_files": [
                    {
                        "split": "train",
                        "url": "hf://datasets/unit/test@refs/convert/parquet/default/train/0000.parquet",
                        "filename": "0000.parquet",
                        "config": "default",
                    }
                ]
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("cr_train.data.source.request_json", initial_request_json)
    first_root, first_descriptor = ensure_source_root(
        dataset_name="unit/test",
        revision=None,
        cache_root=tmp_path,
    )

    _source_descriptor_cache.clear()
    monkeypatch.setattr("cr_train.data.source.request_json", lambda _url: (_ for _ in ()).throw(RuntimeError("offline")))
    second_root, second_descriptor = ensure_source_root(
        dataset_name="unit/test",
        revision=None,
        cache_root=tmp_path,
    )

    assert second_root == first_root
    assert second_descriptor["source_signature"] == first_descriptor["source_signature"]


def test_ensure_split_cache_reuses_cached_rows_across_plans(monkeypatch, tmp_path: Path) -> None:
    rows = [_make_row(index) for index in range(4 * BLOCK_SIZE)]
    split_rows = {"train": rows, "validation": rows[: 2 * BLOCK_SIZE], "test": rows[: 2 * BLOCK_SIZE]}
    stats = _patch_source(monkeypatch, split_rows)

    catalog = _catalog(total_rows=len(rows))
    first_seed = 7
    first_plan = plan_sample(catalog, seed=first_seed, max_samples=2 * BLOCK_SIZE, split="train")
    second_seed, second_plan = next(
        (seed, plan)
        for seed in range(8, 4096)
        for plan in [plan_sample(catalog, seed=seed, max_samples=2 * BLOCK_SIZE, split="train")]
        if 0 < len(set(first_plan.selected_row_ids.tolist()) & set(plan.selected_row_ids.tolist())) < len(plan.selected_row_ids)
    )

    catalog_path = ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=first_seed,
        cache_root=tmp_path,
    )
    store_root = _row_store_root(catalog_path, "train")
    cached_after_first = np.load(store_root / "cached_rows.npy")

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=second_seed,
        cache_root=tmp_path,
    )
    cached_after_second = np.load(store_root / "cached_rows.npy")

    first_rows = set(int(value) for value in first_plan.selected_row_ids.tolist())
    second_rows = set(int(value) for value in second_plan.selected_row_ids.tolist())

    assert stats["request_json_calls"] == 2
    assert int(np.count_nonzero(cached_after_first)) == len(first_rows)
    assert int(np.count_nonzero(cached_after_second)) == len(first_rows | second_rows)
    assert len(first_rows & second_rows) > 0
    assert len(first_rows | second_rows) < (len(first_rows) + len(second_rows))


def test_ensure_split_cache_skips_remote_read_when_selection_is_fully_cached(monkeypatch, tmp_path: Path) -> None:
    rows = [_make_row(index) for index in range(4 * BLOCK_SIZE)]
    split_rows = {"train": rows, "validation": rows[: BLOCK_SIZE], "test": rows[: BLOCK_SIZE]}
    startup_events: list[dict[str, object]] = []

    _FakeTqdm.instances.clear()
    monkeypatch.setattr("cr_train.data.runtime.tqdm", _FakeTqdm)
    stats = _patch_source(monkeypatch, split_rows)

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=3,
        cache_root=tmp_path,
        startup_callback=lambda event: startup_events.append(dict(event)),
    )
    read_calls_after_first = list(stats["read_row_groups"])
    warmup_bars_after_first = len(_FakeTqdm.instances)

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=3,
        cache_root=tmp_path,
        startup_callback=lambda event: startup_events.append(dict(event)),
    )

    done_events = [
        event
        for event in startup_events
        if event["stage"] == "warm split cache" and event["status"] == "done"
    ]

    assert stats["request_json_calls"] == 2
    assert stats["read_row_groups"] == read_calls_after_first
    assert len(_FakeTqdm.instances) == warmup_bars_after_first
    assert done_events[-1]["cache_only"] is True
    assert done_events[-1]["selected_missing_blocks"] == 0
    assert done_events[-1]["resolved_blocks"] == 0
    assert "frontier_before" not in done_events[-1]
    assert "extension_blocks" not in done_events[-1]


def test_warmup_progress_tracks_actual_missing_blocks(monkeypatch, tmp_path: Path) -> None:
    rows = [_make_row(index) for index in range(10 * BLOCK_SIZE)]
    split_rows = {"train": rows, "validation": rows[: BLOCK_SIZE], "test": rows[: BLOCK_SIZE]}
    startup_events: list[dict[str, object]] = []

    _FakeTqdm.instances.clear()
    monkeypatch.setattr("cr_train.data.runtime.tqdm", _FakeTqdm)
    stats = _patch_source(monkeypatch, split_rows)

    seed, sample_plan = next(
        (seed, plan)
        for seed in range(4096)
        for plan in [plan_sample(_catalog(total_rows=len(rows)), seed=seed, max_samples=2 * BLOCK_SIZE, split="train")]
        if plan.execution_block_count > plan.required_blocks
    )

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=seed,
        cache_root=tmp_path,
        startup_callback=lambda event: startup_events.append(dict(event)),
    )

    done_event = next(
        event
        for event in startup_events
        if event["stage"] == "warm split cache" and event["status"] == "done"
    )
    progress = _FakeTqdm.instances[-1]

    assert done_event["execution_block_count"] == sample_plan.execution_block_count
    assert done_event["selected_missing_blocks"] == sample_plan.required_blocks
    assert done_event["execution_block_count"] > done_event["selected_missing_blocks"]
    assert int(progress.total) == done_event["selected_missing_blocks"]
    assert sum(progress.updates) == done_event["selected_missing_blocks"]
    assert progress.desc_history == [str(progress.desc)]
    train_url = "hf://datasets/unit/test@refs/convert/parquet/default/train/0000.parquet"
    assert stats["opens"][train_url] == 2


def test_warmup_download_speed_is_windowed_for_stability(monkeypatch) -> None:
    import cr_train.data.runtime as runtime_mod

    class ManualClock:
        def __init__(self) -> None:
            self.current = 0.0

        def perf_counter(self) -> float:
            return self.current

    clock = ManualClock()
    progress = _FakeTqdm(disable=False)
    state = WarmupProgressState()
    monkeypatch.setattr(runtime_mod.time, "perf_counter", clock.perf_counter)

    _update_warmup_progress(
        progress,
        state=state,
        resolved_blocks=0,
        selected_missing_blocks=2,
        selected_block_count=2,
    )
    clock.current = 0.1
    _update_warmup_progress(
        progress,
        state=state,
        resolved_blocks=0,
        selected_missing_blocks=2,
        selected_block_count=2,
        downloaded_bytes_delta=10 * 1024 * 1024,
    )
    clock.current = 0.2
    _update_warmup_progress(
        progress,
        state=state,
        resolved_blocks=0,
        selected_missing_blocks=2,
        selected_block_count=2,
        downloaded_bytes_delta=10 * 1024 * 1024,
    )
    clock.current = 0.7
    _update_warmup_progress(
        progress,
        state=state,
        resolved_blocks=0,
        selected_missing_blocks=2,
        selected_block_count=2,
        downloaded_bytes_delta=10 * 1024 * 1024,
    )

    assert "0.0 MB/s" in progress.postfixes[0]
    assert "0.0 MB/s" in progress.postfixes[1]
    assert "0.0 MB/s" in progress.postfixes[2]
    assert "50.0 MB/s" in progress.postfixes[3]


def test_prepare_split_reads_cached_rows_in_selected_logical_block_order(monkeypatch, tmp_path: Path) -> None:
    rows = [_make_row(index) for index in range(4 * BLOCK_SIZE)]
    split_rows = {"train": rows, "validation": rows, "test": rows}
    _patch_source(monkeypatch, split_rows)
    monkeypatch.setattr("cr_train.data.runtime.tqdm", _FakeTqdm)

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
        _catalog(total_rows=len(rows)),
        seed=13,
        max_samples=2 * BLOCK_SIZE,
        split="validation",
    )

    assert batch_scenes == [str(row_id) for row_id in sample_plan.selected_row_ids.tolist()]
