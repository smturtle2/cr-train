from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from cr_train.data import (
    BLOCK_SIZE,
    CANONICAL_SHUFFLE_BUFFER_SIZE,
    build_collate_fn,
    build_dataloader,
    compress_execution_runs,
    compute_base_take_probability,
    compute_take_probability,
    decode_row,
    ensure_split_cache,
    normalize_parquet_uri,
    plan_sample,
    prepare_split,
)
from cr_train.data.runtime import _compact_warmup_timeline, _render_warmup_timeline


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
        self.postfixes: list[dict[str, object]] = []
        self.desc = kwargs.get("desc")
        self.desc_history: list[str] = [str(self.desc)] if self.desc is not None else []
        _FakeTqdm.instances.append(self)

    @staticmethod
    def write(_message: str) -> None:
        return None

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
        "load_dataset_calls": [],
        "opens": defaultdict(int),
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
        return _FakeParquetFile(
            str(url),
            row_group_rows=list(shard_rows[str(url)]),
            stats=stats,
        )

    def fake_load_dataset(path: str, *args, split: str | None = None, streaming: bool = False, **kwargs):
        del args, kwargs
        assert path == "unit/test"
        assert streaming is True
        assert split is not None
        stats["load_dataset_calls"].append(str(split))
        return _FakeStreamingDataset(list(split_rows[str(split)]))

    monkeypatch.setattr("cr_train.data.source.request_json", fake_request_json)
    monkeypatch.setattr("cr_train.data.source.pq.ParquetFile", fake_parquet_file)
    monkeypatch.setattr("cr_train.data.runtime.load_dataset", fake_load_dataset)
    return stats


def _catalog(total_rows: int) -> dict[str, object]:
    return {"total_rows": total_rows}


def _uniform_execution_block_count(seed: int, *, required_blocks: int, candidate_blocks: int) -> int:
    rng = np.random.default_rng(seed)
    selected_blocks = np.sort(
        rng.choice(candidate_blocks, size=required_blocks, replace=False).astype(np.int64)
    )
    return int(selected_blocks[-1]) + 1


def _cache_store_root(catalog_path: Path, split: str, dataset_seed: int) -> Path:
    source_root = catalog_path.parent.parent
    return (
        source_root
        / "block_store"
        / split
        / f"dataset-seed={dataset_seed}-shuffle-buffer={CANONICAL_SHUFFLE_BUFFER_SIZE}"
    )


def _canonical_rows(rows: list[dict[str, object]], dataset_seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(dataset_seed)
    order = rng.permutation(len(rows))
    return [rows[int(index)] for index in order]


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


def testplan_sample_is_block_reproducible_and_candidate_bounded() -> None:
    catalog = _catalog(total_rows=10 * BLOCK_SIZE)

    sample_a = plan_sample(catalog, seed=7, max_samples=20)
    sample_b = plan_sample(catalog, seed=7, max_samples=20)
    sample_c = plan_sample(catalog, seed=99, max_samples=20)

    assert sample_a.requested_rows == 20
    assert sample_a.required_blocks == 2
    assert sample_a.effective_rows == 2 * BLOCK_SIZE
    assert sample_a.total_blocks == 10
    assert sample_a.candidate_blocks == 4
    assert sample_a.planner_mode == "sequential_additive_exact_k"
    assert sample_a.base_take_prob == pytest.approx(0.5)
    assert sample_a.selected_blocks.size == sample_a.required_blocks
    assert np.array_equal(sample_a.selected_blocks, sample_b.selected_blocks)
    assert not np.array_equal(sample_a.selected_blocks, sample_c.selected_blocks)
    assert np.all(sample_a.selected_blocks < sample_a.candidate_blocks)
    assert sample_a.execution_block_count == int(sample_a.selected_blocks[-1]) + 1


def test_take_probability_grows_as_candidate_suffix_shrinks() -> None:
    base_take_prob = compute_base_take_probability(4, 16)
    take_probs = [
        compute_take_probability(4, 16, remaining_candidates)
        for remaining_candidates in range(16, 3, -1)
    ]

    assert base_take_prob == pytest.approx(0.25)
    assert take_probs[0] == pytest.approx(base_take_prob)
    assert take_probs[-1] == pytest.approx(1.0)
    assert take_probs == sorted(take_probs)


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


def testplan_sample_suffix_guard_preserves_exact_block_count(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NeverTakeRng:
        @staticmethod
        def random() -> float:
            return 1.0

    monkeypatch.setattr("cr_train.data.planning.np.random.default_rng", lambda _seed: _NeverTakeRng())

    sample_plan = plan_sample(_catalog(total_rows=10 * BLOCK_SIZE), seed=7, max_samples=2 * BLOCK_SIZE)

    assert sample_plan.selected_blocks.tolist() == [2, 3]
    assert sample_plan.selected_blocks.size == sample_plan.required_blocks


def testplan_sample_reduces_execution_span_vs_uniform_choice_baseline() -> None:
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
                candidate_blocks=sample_plan.candidate_blocks,
            )
        )

    assert float(np.mean(sequential_counts)) < float(np.mean(uniform_counts))


def testcompress_execution_runs_merges_adjacent_block_spans() -> None:
    selected = np.asarray([0, 0, 1, 1, 1, 1, 0, 0, 1], dtype=np.bool_)
    cached = np.asarray([0, 0, 1, 1, 0, 0, 0, 0, 1], dtype=np.bool_)

    runs = compress_execution_runs(selected, cached, stop_block=9)

    assert [(run.kind, run.total_rows) for run in runs] == [
        ("skip", 2 * BLOCK_SIZE),
        ("take_cached", 2 * BLOCK_SIZE),
        ("take_remote", 2 * BLOCK_SIZE),
        ("skip", 2 * BLOCK_SIZE),
        ("take_cached", BLOCK_SIZE),
    ]


def testnormalize_parquet_uri_converts_hf_resolve_url() -> None:
    url = "https://huggingface.co/datasets/Hermanni/sen12mscr/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"

    assert normalize_parquet_uri(url) == "hf://datasets/Hermanni/sen12mscr@refs/convert/parquet/default/train/0000.parquet"


def test_ensure_split_cache_rewinds_for_missing_blocks_before_frontier(monkeypatch, tmp_path: Path) -> None:
    rows = [_make_row(index) for index in range(4 * BLOCK_SIZE)]
    split_rows = {
        "train": rows,
        "validation": rows[: 2 * BLOCK_SIZE],
        "test": rows[: 2 * BLOCK_SIZE],
    }
    stats = _patch_source(monkeypatch, split_rows)

    catalog = _catalog(total_rows=len(rows))
    plans = [
        (seed, plan_sample(catalog, seed=seed, max_samples=BLOCK_SIZE, split="train"))
        for seed in range(512)
    ]
    first_seed, first_plan = next(
        (seed, plan)
        for seed, plan in plans
        if plan.execution_block_count >= 2
    )
    second_seed, second_plan = next(
        (seed, plan)
        for seed, plan in plans
        if seed != first_seed
        and any(
            int(block) < first_plan.execution_block_count and int(block) not in first_plan.selected_blocks
            for block in plan.selected_blocks
        )
    )
    first_block = int(first_plan.selected_blocks[-1])
    second_block = next(
        int(block)
        for block in second_plan.selected_blocks
        if int(block) < first_plan.execution_block_count and int(block) not in first_plan.selected_blocks
    )

    catalog_path = ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=BLOCK_SIZE,
        seed=first_seed,
        dataset_seed=11,
        cache_root=tmp_path,
    )
    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=BLOCK_SIZE,
        seed=second_seed,
        dataset_seed=11,
        cache_root=tmp_path,
    )

    store_root = _cache_store_root(catalog_path, "train", 11)
    state = np.load(store_root / "cached.npy")
    state_payload = (store_root / "state.json").read_text(encoding="utf-8")

    assert stats["load_dataset_calls"] == ["train", "train"]
    assert bool(state[first_block])
    assert bool(state[second_block])
    assert '"canonical_frontier_block":' in state_payload


def test_ensure_split_cache_skips_hf_open_when_selection_is_fully_cached(monkeypatch, tmp_path: Path) -> None:
    rows = [_make_row(index) for index in range(4 * BLOCK_SIZE)]
    split_rows = {
        "train": rows,
        "validation": rows[: BLOCK_SIZE],
        "test": rows[: BLOCK_SIZE],
    }
    startup_events: list[dict[str, object]] = []

    _FakeTqdm.instances.clear()
    monkeypatch.setattr("cr_train.data.runtime.tqdm", _FakeTqdm)
    stats = _patch_source(monkeypatch, split_rows)

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=None,
        seed=3,
        dataset_seed=5,
        cache_root=tmp_path,
        startup_callback=lambda event: startup_events.append(dict(event)),
    )
    load_calls_after_first = list(stats["load_dataset_calls"])
    warmup_bars_after_first = len(_FakeTqdm.instances)

    ensure_split_cache(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=BLOCK_SIZE,
        seed=99,
        dataset_seed=5,
        cache_root=tmp_path,
        startup_callback=lambda event: startup_events.append(dict(event)),
    )

    done_events = [
        event
        for event in startup_events
        if event["stage"] == "warm split cache" and event["status"] == "done"
    ]
    request_json_calls_after_first = stats["request_json_calls"]

    assert load_calls_after_first == ["train"]
    assert stats["load_dataset_calls"] == ["train"]
    assert len(_FakeTqdm.instances) == warmup_bars_after_first
    assert done_events[-1]["missing_blocks"] == 0
    assert done_events[-1]["cache_only"] is True
    assert stats["request_json_calls"] == request_json_calls_after_first


def test_prepare_split_reads_cached_rows_in_selected_canonical_block_order(monkeypatch, tmp_path: Path) -> None:
    rows = [_make_row(index) for index in range(4 * BLOCK_SIZE)]
    split_rows = {
        "train": rows,
        "validation": rows,
        "test": rows,
    }
    stats = _patch_source(monkeypatch, split_rows)
    monkeypatch.setattr("cr_train.data.runtime.tqdm", _FakeTqdm)

    ensure_split_cache(
        split="validation",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=13,
        dataset_seed=17,
        cache_root=tmp_path,
    )
    prepared = prepare_split(
        split="validation",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=13,
        dataset_seed=17,
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
    sample_plan = plan_sample(_catalog(total_rows=len(rows)), seed=13, max_samples=2 * BLOCK_SIZE, split="validation")
    canonical_rows = _canonical_rows(rows, dataset_seed=17)
    expected_rows: list[dict[str, object]] = []
    for block_index in sample_plan.selected_blocks:
        start = int(block_index) * BLOCK_SIZE
        stop = start + BLOCK_SIZE
        expected_rows.extend(canonical_rows[start:stop])

    assert stats["load_dataset_calls"] == ["validation"]
    assert batch_scenes == [str(row["scene"]) for row in expected_rows]


def test_predecoded_cache_produces_identical_batch_values(monkeypatch, tmp_path: Path) -> None:
    """predecoded=True 캐시로부터 생성된 배치가 일반 캐시와 동일한 텐서 값을 갖는지 확인."""
    rows = [_make_row(index) for index in range(2 * BLOCK_SIZE)]
    split_rows = {"train": rows, "validation": rows, "test": rows}
    _patch_source(monkeypatch, split_rows)
    monkeypatch.setattr("cr_train.data.runtime.tqdm", _FakeTqdm)

    common = dict(
        split="train", dataset_name="unit/test", revision=None,
        max_samples=BLOCK_SIZE, seed=7, dataset_seed=3,
    )

    # 일반 캐시
    ensure_split_cache(**common, cache_root=tmp_path / "normal")
    normal_prepared = prepare_split(**common, cache_root=tmp_path / "normal")
    normal_loader = build_dataloader(normal_prepared, batch_size=BLOCK_SIZE, num_workers=0, training=False, seed=7, epoch=0)
    normal_batch = next(iter(normal_loader))

    # predecoded 캐시
    ensure_split_cache(**common, cache_root=tmp_path / "predec", predecoded=True)
    predec_prepared = prepare_split(**common, cache_root=tmp_path / "predec", predecoded=True)
    predec_loader = build_dataloader(predec_prepared, batch_size=BLOCK_SIZE, num_workers=0, training=False, seed=7, epoch=0, predecoded=True)
    predec_batch = next(iter(predec_loader))

    import torch
    assert torch.allclose(normal_batch["sar"], predec_batch["sar"], atol=1e-6)
    assert torch.allclose(normal_batch["cloudy"], predec_batch["cloudy"], atol=1e-6)
    assert torch.allclose(normal_batch["target"], predec_batch["target"], atol=1e-6)
    assert normal_batch["meta"]["scene"] == predec_batch["meta"]["scene"]
