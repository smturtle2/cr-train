from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from datasets import IterableDataset

from cr_train.take_cache import _take_from_prefix_rows, take_rows_official, take_rows_with_prefix_cache


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


class _FakeRecordBatch:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_pylist(self) -> list[dict[str, object]]:
        return list(self._rows)


class _FakeParquetFile:
    def __init__(
        self,
        url: str,
        rows: list[dict[str, object]],
        row_group_rows: list[int],
        stats: dict[str, object],
    ) -> None:
        self._url = url
        self._rows = rows
        self._row_group_rows = row_group_rows
        self.metadata = _FakeParquetMetadata(row_group_rows)
        stats["opens"][url] += 1
        self._stats = stats

    def iter_batches(self, *, row_groups=None, columns=None, batch_size=None, use_threads=False):
        del use_threads
        selected_group_ids = tuple(int(group_id) for group_id in (row_groups or range(len(self._row_group_rows))))
        selected_columns = tuple(str(column) for column in (columns or ()))
        self._stats["iter_batches"].append(
            {
                "url": self._url,
                "row_groups": selected_group_ids,
                "batch_size": int(batch_size or 0),
            }
        )

        row_group_starts = [0]
        for size in self._row_group_rows[:-1]:
            row_group_starts.append(row_group_starts[-1] + size)

        selected_rows: list[dict[str, object]] = []
        for row_group_id in selected_group_ids:
            start = row_group_starts[row_group_id]
            stop = start + self._row_group_rows[row_group_id]
            for row in self._rows[start:stop]:
                selected_rows.append({key: row[key] for key in selected_columns})

        chunk_size = int(batch_size or max(1, len(selected_rows)))
        for start in range(0, len(selected_rows), chunk_size):
            yield _FakeRecordBatch(selected_rows[start:start + chunk_size])


def _make_simple_row(index: int) -> dict[str, object]:
    return {
        "sar": f"sar-{index}".encode("utf-8"),
        "cloudy": f"cloudy-{index}".encode("utf-8"),
        "target": f"target-{index}".encode("utf-8"),
        "sar_shape": [1, 1, 1],
        "opt_shape": [1, 1, 1],
        "season": "spring",
        "scene": str(index),
        "patch": f"p{index:03d}",
    }


def _patch_take_source(monkeypatch, split_shards: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    parquet_entries: list[dict[str, object]] = []
    shard_map: dict[str, dict[str, object]] = {}
    split_rows: dict[str, list[dict[str, object]]] = {}
    for split, shards in split_shards.items():
        split_rows[split] = []
        for index, shard in enumerate(shards):
            url = f"hf://datasets/unit/test@refs/convert/parquet/default/{split}/{index:04d}.parquet"
            parquet_entries.append(
                {
                    "dataset": "unit/test",
                    "config": "default",
                    "split": split,
                    "url": url,
                    "filename": f"{index:04d}.parquet",
                }
            )
            shard_map[url] = {
                "rows": list(shard["rows"]),
                "row_group_rows": list(shard["row_group_rows"]),
            }
            split_rows[split].extend(shard["rows"])

    def fake_request_json(url: str):
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

    stats: dict[str, object] = {
        "opens": defaultdict(int),
        "iter_batches": [],
    }

    def fake_parquet_file(url: str):
        shard = shard_map[str(url)]
        return _FakeParquetFile(
            str(url),
            rows=list(shard["rows"]),
            row_group_rows=list(shard["row_group_rows"]),
            stats=stats,
        )

    def fake_load_dataset(path: str, *args, split: str | None = None, streaming: bool = False, **kwargs):
        assert path == "unit/test"
        assert streaming is True
        assert split is not None

        def generator():
            for row in split_rows[str(split)]:
                yield dict(row)

        return IterableDataset.from_generator(generator)

    monkeypatch.setattr("cr_train.data.source.request_json", fake_request_json)
    monkeypatch.setattr("cr_train.data.source.pq.ParquetFile", fake_parquet_file)
    monkeypatch.setattr("cr_train.take_cache.pq.ParquetFile", fake_parquet_file)
    monkeypatch.setattr("cr_train.take_cache.load_dataset", fake_load_dataset)
    return stats


def test_take_from_prefix_rows_matches_official_shuffle_take() -> None:
    buffer_size = 10
    sample_size = 20
    prefix_rows = [{"id": index} for index in range(buffer_size + sample_size)]

    def generator():
        for index in range(100):
            yield {"id": index}

    baseline = [row["id"] for row in IterableDataset.from_generator(generator).shuffle(seed=42, buffer_size=buffer_size).take(sample_size)]
    replayed = [row["id"] for row in _take_from_prefix_rows(prefix_rows, seed=42, sample_size=sample_size, buffer_size=buffer_size)]

    assert replayed == baseline


def test_prefix_cache_reuses_across_seeds_and_extends_only_the_tail(monkeypatch, tmp_path: Path) -> None:
    rows = [_make_simple_row(index) for index in range(12)]
    split_shards = {
        "train": [
            {"rows": rows[:4], "row_group_rows": [2, 2]},
            {"rows": rows[4:8], "row_group_rows": [2, 2]},
            {"rows": rows[8:], "row_group_rows": [2, 2]},
        ],
        "validation": [{"rows": rows[:4], "row_group_rows": [2, 2]}],
        "test": [{"rows": rows[:4], "row_group_rows": [2, 2]}],
    }
    stats = _patch_take_source(monkeypatch, split_shards)

    first_official = take_rows_official(
        split="train",
        seed=42,
        sample_size=3,
        buffer_size=4,
        dataset_name="unit/test",
        cache_dir=tmp_path / "official",
    )
    first_cached = take_rows_with_prefix_cache(
        split="train",
        seed=42,
        sample_size=3,
        buffer_size=4,
        dataset_name="unit/test",
        cache_dir=tmp_path / "prototype",
    )
    assert first_cached.digest == first_official.digest
    assert first_cached.cache_stats is not None
    assert first_cached.cache_stats.fetched_prefix_rows == 7

    iter_batch_calls_after_first = len(stats["iter_batches"])
    second_official = take_rows_official(
        split="train",
        seed=99,
        sample_size=3,
        buffer_size=4,
        dataset_name="unit/test",
        cache_dir=tmp_path / "official",
    )
    second_cached = take_rows_with_prefix_cache(
        split="train",
        seed=99,
        sample_size=3,
        buffer_size=4,
        dataset_name="unit/test",
        cache_dir=tmp_path / "prototype",
    )
    assert second_cached.digest == second_official.digest
    assert second_cached.cache_stats is not None
    assert second_cached.cache_stats.fetched_prefix_rows == 0
    assert len(stats["iter_batches"]) == iter_batch_calls_after_first

    third_official = take_rows_official(
        split="train",
        seed=99,
        sample_size=5,
        buffer_size=4,
        dataset_name="unit/test",
        cache_dir=tmp_path / "official",
    )
    third_cached = take_rows_with_prefix_cache(
        split="train",
        seed=99,
        sample_size=5,
        buffer_size=4,
        dataset_name="unit/test",
        cache_dir=tmp_path / "prototype",
    )
    assert third_cached.digest == third_official.digest
    assert third_cached.cache_stats is not None
    assert third_cached.cache_stats.fetched_prefix_rows == 2
