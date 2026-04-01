from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import IterableDataset, load_dataset

from .data.constants import DATA_COLUMNS, DATASET_ID
from .data.source import ensure_source_root, ensure_split_catalog, resolve_source_root
from .data.store import (
    file_lock,
    freeze_row,
    read_json,
    remove_tree,
    resolve_cache_root,
    write_json_atomic,
)

PREFIX_CACHE_CHUNK_ROWS = 128


@dataclass(slots=True)
class PrefixCacheState:
    prefix_rows: int
    chunk_row_counts: list[int]


@dataclass(slots=True)
class PrefixCacheStats:
    """prefix 캐시 보장 작업의 통계 (캐시 히트/미스, 소요 시간 등)."""

    source_signature: str
    required_prefix_rows: int
    cached_prefix_rows_before: int
    cached_prefix_rows_after: int
    fetched_prefix_rows: int
    fetched_shards: int
    elapsed_sec: float


@dataclass(slots=True)
class SampleResult:
    """행 샘플링 결과. digest로 결정적 재현성 검증 가능."""

    rows: list[dict[str, Any]]
    digest: str
    elapsed_sec: float
    cache_stats: PrefixCacheStats | None


def _resolve_take_root(source_root: Path) -> Path:
    path = source_root / "prefix_store"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_take_split_root(source_root: Path, split: str) -> Path:
    path = _resolve_take_root(source_root) / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_take_lock_path(source_root: Path, split: str) -> Path:
    return _resolve_take_split_root(source_root, split) / ".lock"


def _resolve_take_chunks_root(source_root: Path, split: str) -> Path:
    path = _resolve_take_split_root(source_root, split) / "chunks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_take_state_path(source_root: Path, split: str) -> Path:
    return _resolve_take_split_root(source_root, split) / "prefix.json"


def _chunk_path(chunks_root: Path, chunk_index: int) -> Path:
    return chunks_root / f"{chunk_index:08d}.arrow"


def _load_or_init_prefix_state(
    *,
    source_root: Path,
    split: str,
) -> PrefixCacheState:
    state_path = _resolve_take_state_path(source_root, split)
    if not state_path.exists():
        state = PrefixCacheState(prefix_rows=0, chunk_row_counts=[])
        _write_prefix_state(
            source_root=source_root,
            split=split,
            state=state,
        )
        return state

    payload = read_json(state_path)
    return PrefixCacheState(
        prefix_rows=int(payload["prefix_rows"]),
        chunk_row_counts=[int(value) for value in payload["chunk_row_counts"]],
    )


def _write_prefix_state(
    *,
    source_root: Path,
    split: str,
    state: PrefixCacheState,
) -> None:
    write_json_atomic(
        _resolve_take_state_path(source_root, split),
        {
            "prefix_rows": state.prefix_rows,
            "chunk_row_counts": state.chunk_row_counts,
        },
    )


def _save_chunk(
    *,
    source_root: Path,
    split: str,
    chunk_index: int,
    rows: list[dict[str, Any]],
) -> int:
    chunks_root = _resolve_take_chunks_root(source_root, split)
    chunk_file = _chunk_path(chunks_root, chunk_index)
    tmp_file = chunk_file.with_suffix(".tmp")
    remove_tree(tmp_file)
    table = pa.table({col: [row[col] for row in rows] for col in DATA_COLUMNS})
    with pa.OSFile(str(tmp_file), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table)
        writer.close()
    tmp_file.replace(chunk_file)
    return len(rows)


def _iter_rows_from_prefix_range(
    catalog: Mapping[str, Any],
    *,
    start_row: int,
    stop_row: int,
) -> Iterator[dict[str, Any]]:
    if start_row >= stop_row:
        return

    for shard in catalog["shards"]:
        shard_start = int(shard["global_start"])
        shard_stop = int(shard["global_stop"])
        overlap_start = max(start_row, shard_start)
        overlap_stop = min(stop_row, shard_stop)
        if overlap_start >= overlap_stop:
            continue

        local_start = overlap_start - shard_start
        local_stop = overlap_stop - shard_start
        row_group_rows = [int(value) for value in shard["row_group_rows"]]
        row_group_starts: list[int] = []
        current = 0
        for size in row_group_rows:
            row_group_starts.append(current)
            current += size
        row_group_ends = [start + size for start, size in zip(row_group_starts, row_group_rows)]
        first_group = next(index for index, end in enumerate(row_group_ends) if local_start < end)
        last_group = next(index for index, end in enumerate(row_group_ends) if (local_stop - 1) < end)
        parquet_file = pq.ParquetFile(str(shard["url"]))
        rows_to_skip = local_start - row_group_starts[first_group]
        rows_to_take = overlap_stop - overlap_start
        rows_taken = 0
        rows_seen = 0
        for batch in parquet_file.iter_batches(
            row_groups=list(range(first_group, last_group + 1)),
            columns=DATA_COLUMNS,
            batch_size=PREFIX_CACHE_CHUNK_ROWS,
            use_threads=False,
        ):
            batch_rows = batch.to_pylist()
            batch_start = rows_seen
            batch_stop = rows_seen + len(batch_rows)
            slice_start = max(rows_to_skip, batch_start)
            slice_stop = min(rows_to_skip + rows_to_take, batch_stop)
            if slice_start < slice_stop:
                for row in batch_rows[slice_start - batch_start:slice_stop - batch_start]:
                    yield freeze_row(row)
                    rows_taken += 1
                    if rows_taken >= rows_to_take:
                        break
            rows_seen = batch_stop
            if rows_taken >= rows_to_take:
                break


def _append_prefix_rows(
    *,
    source_root: Path,
    split: str,
    rows: Iterator[dict[str, Any]],
    state: PrefixCacheState,
) -> int:
    chunk_index = len(state.chunk_row_counts)
    pending_rows: list[dict[str, Any]] = []
    appended_rows = 0
    for row in rows:
        pending_rows.append(row)
        if len(pending_rows) < PREFIX_CACHE_CHUNK_ROWS:
            continue
        written = _save_chunk(
            source_root=source_root,
            split=split,
            chunk_index=chunk_index,
            rows=pending_rows,
        )
        state.chunk_row_counts.append(written)
        state.prefix_rows += written
        appended_rows += written
        chunk_index += 1
        pending_rows = []

    if pending_rows:
        written = _save_chunk(
            source_root=source_root,
            split=split,
            chunk_index=chunk_index,
            rows=pending_rows,
        )
        state.chunk_row_counts.append(written)
        state.prefix_rows += written
        appended_rows += written

    return appended_rows


def _ensure_prefix_cache(
    *,
    split: str,
    dataset_name: str,
    revision: str | None,
    required_prefix_rows: int,
    cache_root: Path,
) -> PrefixCacheStats:
    started_at = time.perf_counter()
    source_root, descriptor = ensure_source_root(
        dataset_name=dataset_name,
        revision=revision,
        cache_root=cache_root,
    )
    source_signature = str(descriptor["source_signature"])
    with file_lock(_resolve_take_lock_path(source_root, split)):
        catalog = ensure_split_catalog(
            source_root=source_root,
            descriptor=descriptor,
            split=split,
            startup_callback=None,
        )
        total_rows = int(catalog["total_rows"])
        capped_prefix_rows = min(required_prefix_rows, total_rows)
        state = _load_or_init_prefix_state(
            source_root=source_root,
            split=split,
        )
        before = state.prefix_rows
        if state.prefix_rows < capped_prefix_rows:
            _append_prefix_rows(
                source_root=source_root,
                split=split,
                rows=_iter_rows_from_prefix_range(
                    catalog,
                    start_row=state.prefix_rows,
                    stop_row=capped_prefix_rows,
                ),
                state=state,
            )
            _write_prefix_state(
                source_root=source_root,
                split=split,
                state=state,
            )

        fetched_rows = max(0, state.prefix_rows - before)
        fetched_shards = 0
        if fetched_rows > 0:
            for shard in catalog["shards"]:
                overlap_start = max(before, int(shard["global_start"]))
                overlap_stop = min(state.prefix_rows, int(shard["global_stop"]))
                if overlap_start < overlap_stop:
                    fetched_shards += 1

        return PrefixCacheStats(
            source_signature=source_signature,
            required_prefix_rows=capped_prefix_rows,
            cached_prefix_rows_before=before,
            cached_prefix_rows_after=state.prefix_rows,
            fetched_prefix_rows=fetched_rows,
            fetched_shards=fetched_shards,
            elapsed_sec=time.perf_counter() - started_at,
        )


def _iter_cached_prefix_rows(
    *,
    cache_root: Path,
    source_signature: str,
    split: str,
    limit: int,
) -> Iterator[dict[str, Any]]:
    source_root = resolve_source_root(cache_root, source_signature)
    state = _load_or_init_prefix_state(
        source_root=source_root,
        split=split,
    )
    if limit > state.prefix_rows:
        raise ValueError(f"requested prefix {limit} exceeds cached prefix {state.prefix_rows}")

    chunks_root = _resolve_take_chunks_root(source_root, split)
    consumed = 0
    for chunk_index, chunk_row_count in enumerate(state.chunk_row_counts):
        if consumed >= limit:
            break
        table = pa.ipc.open_file(pa.memory_map(str(_chunk_path(chunks_root, chunk_index)), "r")).read_all()
        take_rows = min(int(chunk_row_count), limit - consumed)
        for row_index in range(take_rows):
            yield {col: table.column(col)[row_index].as_py() for col in DATA_COLUMNS}
        consumed += take_rows


def _take_from_cached_prefix(
    *,
    cache_root: Path,
    source_signature: str,
    split: str,
    limit: int,
    seed: int,
    sample_size: int,
    buffer_size: int,
) -> list[dict[str, Any]]:
    def generator(
        cache_root_str: str,
        source_sig: str,
        split_name: str,
        prefix_limit: int,
    ) -> Iterator[dict[str, Any]]:
        yield from _iter_cached_prefix_rows(
            cache_root=Path(cache_root_str),
            source_signature=source_sig,
            split=split_name,
            limit=prefix_limit,
        )

    dataset = IterableDataset.from_generator(
        generator,
        gen_kwargs={
            "cache_root_str": str(cache_root),
            "source_sig": source_signature,
            "split_name": split,
            "prefix_limit": limit,
        },
    )
    return list(dataset.shuffle(seed=seed, buffer_size=buffer_size).take(sample_size))


def digest_rows(rows: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for column in DATA_COLUMNS:
            value = row[column]
            if isinstance(value, bytes):
                payload = value
            else:
                payload = json.dumps(value, sort_keys=True).encode("utf-8")
            digest.update(column.encode("utf-8"))
            digest.update(b"\0")
            digest.update(payload)
            digest.update(b"\1")
    return digest.hexdigest()


def take_rows_official(
    *,
    split: str,
    seed: int,
    sample_size: int,
    buffer_size: int,
    dataset_name: str = DATASET_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> SampleResult:
    """HF 공식 streaming shuffle-take 파이프라인으로 행 샘플링."""
    started_at = time.perf_counter()
    cache_root = resolve_cache_root(cache_dir)
    dataset = load_dataset(
        dataset_name,
        split=split,
        revision=revision,
        streaming=True,
        cache_dir=str(cache_root),
    ).select_columns(DATA_COLUMNS)
    rows = [dict(row) for row in dataset.shuffle(seed=seed, buffer_size=buffer_size).take(sample_size)]
    return SampleResult(
        rows=rows,
        digest=digest_rows(rows),
        elapsed_sec=time.perf_counter() - started_at,
        cache_stats=None,
    )


def take_rows_with_prefix_cache(
    *,
    split: str,
    seed: int,
    sample_size: int,
    buffer_size: int,
    dataset_name: str = DATASET_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> SampleResult:
    """로컬 prefix 캐시를 활용한 결정적 행 샘플링. 캐시 미스 시 자동으로 prefix를 확장."""
    started_at = time.perf_counter()
    cache_root = resolve_cache_root(cache_dir)
    required_prefix_rows = buffer_size + sample_size
    cache_stats = _ensure_prefix_cache(
        split=split,
        dataset_name=dataset_name,
        revision=revision,
        required_prefix_rows=required_prefix_rows,
        cache_root=cache_root,
    )
    sampled_rows = _take_from_cached_prefix(
        cache_root=cache_root,
        source_signature=cache_stats.source_signature,
        split=split,
        limit=cache_stats.required_prefix_rows,
        seed=seed,
        sample_size=sample_size,
        buffer_size=buffer_size,
    )
    return SampleResult(
        rows=sampled_rows,
        digest=digest_rows(sampled_rows),
        elapsed_sec=time.perf_counter() - started_at,
        cache_stats=cache_stats,
    )
