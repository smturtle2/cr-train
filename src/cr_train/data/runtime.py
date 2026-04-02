from __future__ import annotations

import bisect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .constants import (
    BLOCK_SIZE,
    CHUNK_TARGET_BLOCKS,
    DATA_COLUMNS,
    WARMUP_DOWNLOAD_SPEED_WINDOW_SEC,
    WARMUP_SPEED_EMA_ALPHA,
    WARMUP_TIMELINE_WIDTH,
)
from .planning import SamplePlan, plan_sample
from .source import emit_startup_event, ensure_source_root, ensure_split_catalog, resolve_catalog_path
from .store import (
    SplitRowCache,
    file_lock,
    freeze_row,
    load_or_init_row_cache,
    materialize_rows,
    resolve_row_cache_paths,
    write_row_cache,
)


@dataclass(slots=True)
class WarmupProgressState:
    """EMA state for missing logical-block ingest speed."""

    ema_blocks_per_sec: float | None = None
    last_block_update_at: float | None = None
    last_resolved_blocks: int = 0
    ema_download_bytes_per_sec: float | None = None
    pending_download_bytes: int = 0
    download_window_started_at: float | None = None


@dataclass(slots=True)
class WarmupSummary:
    requested_rows: int
    effective_rows: int
    required_blocks: int
    planner_mode: str
    stop_bias_alpha: float
    selected_block_count: int
    cached_selected_blocks: int
    selected_missing_blocks: int
    cache_only: bool
    execution_block_count: int
    resolved_blocks: int = 0

    def event_fields(self) -> dict[str, Any]:
        return {
            "requested_rows": self.requested_rows,
            "effective_rows": self.effective_rows,
            "required_blocks": self.required_blocks,
            "planner_mode": self.planner_mode,
            "stop_bias_alpha": self.stop_bias_alpha,
            "selected_block_count": self.selected_block_count,
            "cached_selected_blocks": self.cached_selected_blocks,
            "selected_missing_blocks": self.selected_missing_blocks,
            "cache_only": self.cache_only,
            "execution_block_count": self.execution_block_count,
            "resolved_blocks": self.resolved_blocks,
        }


def _emit_warmup_summary(
    startup_callback,
    *,
    split: str,
    status: str,
    summary: WarmupSummary,
    elapsed_sec: float | None = None,
    timeline: str | None = None,
) -> None:
    event = {
        "stage": "warm split cache",
        "split": split,
        "status": status,
        **summary.event_fields(),
    }
    if elapsed_sec is not None:
        event["elapsed_sec"] = elapsed_sec
    if timeline is not None:
        event["timeline"] = timeline
    emit_startup_event(startup_callback, **event)


def _render_warmup_timeline(selected_bitmap, *, stop_block: int) -> str:
    if stop_block <= 0:
        return ""
    return "".join("█" if selected_bitmap[i] else "░" for i in range(stop_block))


def _compact_warmup_timeline(
    timeline: str,
    *,
    max_chars: int = WARMUP_TIMELINE_WIDTH,
) -> str:
    if len(timeline) <= max_chars:
        return timeline
    if max_chars <= 1:
        return timeline[:max_chars]
    head_chars = max(1, (max_chars - 1) // 2)
    tail_chars = max(1, max_chars - head_chars - 1)
    return f"{timeline[:head_chars]}…{timeline[-tail_chars:]}"


def _format_rate(value: float, unit: str) -> str:
    return f"{value:.1f} {unit}" if value < 100 else f"{value:.0f} {unit}"


def _set_progress_postfix_str(progress: Any, text: str) -> None:
    if hasattr(progress, "set_postfix_str"):
        progress.set_postfix_str(text)
        return
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(text)


def _update_warmup_progress(
    progress: Any,
    *,
    state: WarmupProgressState,
    resolved_blocks: int,
    selected_missing_blocks: int,
    selected_block_count: int,
    downloaded_bytes_delta: int = 0,
    force: bool = False,
) -> None:
    if getattr(progress, "disable", False):
        return

    now = time.perf_counter()
    if state.last_block_update_at is None:
        state.last_block_update_at = now
        state.last_resolved_blocks = resolved_blocks
    else:
        delta_blocks = resolved_blocks - state.last_resolved_blocks
        elapsed = max(now - state.last_block_update_at, 1e-9)
        if delta_blocks > 0:
            instant_blocks_per_sec = delta_blocks / elapsed
            if state.ema_blocks_per_sec is None:
                state.ema_blocks_per_sec = instant_blocks_per_sec
            else:
                state.ema_blocks_per_sec = (
                    (1.0 - WARMUP_SPEED_EMA_ALPHA) * state.ema_blocks_per_sec
                    + WARMUP_SPEED_EMA_ALPHA * instant_blocks_per_sec
                )
            state.last_resolved_blocks = resolved_blocks
            state.last_block_update_at = now
        elif force and state.ema_blocks_per_sec is None:
            state.ema_blocks_per_sec = 0.0

    if downloaded_bytes_delta > 0:
        if state.download_window_started_at is None:
            state.download_window_started_at = now
        state.pending_download_bytes += downloaded_bytes_delta

    if state.pending_download_bytes > 0 and state.download_window_started_at is not None:
        elapsed = now - state.download_window_started_at
        if force or elapsed >= WARMUP_DOWNLOAD_SPEED_WINDOW_SEC:
            instant_download_bytes_per_sec = state.pending_download_bytes / max(elapsed, 1e-9)
            if state.ema_download_bytes_per_sec is None:
                state.ema_download_bytes_per_sec = instant_download_bytes_per_sec
            else:
                state.ema_download_bytes_per_sec = (
                    (1.0 - WARMUP_SPEED_EMA_ALPHA) * state.ema_download_bytes_per_sec
                    + WARMUP_SPEED_EMA_ALPHA * instant_download_bytes_per_sec
                )
            state.pending_download_bytes = 0
            state.download_window_started_at = now
    elif force and state.ema_download_bytes_per_sec is None:
        state.ema_download_bytes_per_sec = 0.0

    block_speed = state.ema_blocks_per_sec or 0.0
    download_speed_mb_per_sec = (state.ema_download_bytes_per_sec or 0.0) / (1024.0 * 1024.0)
    _set_progress_postfix_str(
        progress,
        ", ".join(
            (
                f"sel: {selected_block_count}",
                _format_rate(block_speed, "blk/s"),
                _format_rate(download_speed_mb_per_sec, "MB/s"),
            )
        ),
    )


def _selected_block_missing_mask(sample_plan: SamplePlan, cache: SplitRowCache) -> np.ndarray:
    block_missing = np.zeros(sample_plan.required_blocks, dtype=np.bool_)
    for i in range(sample_plan.required_blocks):
        start = int(sample_plan.selected_row_offsets[i])
        stop = int(sample_plan.selected_row_offsets[i + 1])
        row_ids = sample_plan.selected_row_ids[start:stop]
        block_missing[i] = bool(row_ids.size > 0 and not np.all(cache.cached[row_ids]))
    return block_missing


def _block_row_ids(sample_plan: SamplePlan, index: int) -> np.ndarray:
    start = int(sample_plan.selected_row_offsets[index])
    stop = int(sample_plan.selected_row_offsets[index + 1])
    return sample_plan.selected_row_ids[start:stop]


def _build_warmup_summary(
    sample_plan: SamplePlan,
    *,
    cache: SplitRowCache,
) -> WarmupSummary:
    block_missing_mask = _selected_block_missing_mask(sample_plan, cache)
    selected_missing_blocks = int(np.count_nonzero(block_missing_mask))
    cached_selected_blocks = int(sample_plan.required_blocks - selected_missing_blocks)
    execution_block_count = int(sample_plan.execution_block_count)
    return WarmupSummary(
        requested_rows=sample_plan.requested_rows,
        effective_rows=sample_plan.effective_rows,
        required_blocks=sample_plan.required_blocks,
        planner_mode=sample_plan.planner_mode,
        stop_bias_alpha=sample_plan.stop_bias_alpha,
        selected_block_count=sample_plan.required_blocks,
        cached_selected_blocks=cached_selected_blocks,
        selected_missing_blocks=selected_missing_blocks,
        cache_only=selected_missing_blocks == 0,
        execution_block_count=execution_block_count,
    )


def _row_group_start_offsets(row_group_rows: list[int]) -> list[int]:
    starts: list[int] = []
    current = 0
    for size in row_group_rows:
        starts.append(current)
        current += int(size)
    return starts


def _group_row_ids_by_shard(catalog: dict[str, Any], row_ids: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    if row_ids.size == 0:
        return {}

    shards = list(catalog["shards"])
    shard_stops = [int(shard["global_stop"]) for shard in shards]
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row_id in sorted({int(value) for value in row_ids.tolist()}):
        shard_index = bisect.bisect_right(shard_stops, row_id)
        if shard_index >= len(shards):
            raise IndexError(f"row id {row_id} is out of range for split catalog")
        shard = shards[shard_index]
        local_row = row_id - int(shard["global_start"])
        starts = _row_group_start_offsets([int(value) for value in shard["row_group_rows"]])
        row_group_index = 0
        for index, start in enumerate(starts):
            stop = start + int(shard["row_group_rows"][index])
            if local_row < stop:
                row_group_index = index
                break
        local_row_group_offset = local_row - starts[row_group_index]
        grouped.setdefault((shard_index, row_group_index), []).append((row_id, local_row_group_offset))
    return grouped


def _fetch_rows_by_id(
    catalog: dict[str, Any],
    row_ids: np.ndarray,
    *,
    on_row_group=None,
) -> dict[int, dict[str, Any]]:
    groups = _group_row_ids_by_shard(catalog, row_ids)
    fetched: dict[int, dict[str, Any]] = {}
    shards = list(catalog["shards"])
    groups_by_shard: dict[int, list[tuple[int, list[tuple[int, int]]]]] = {}
    for (shard_index, row_group_index), entries in groups.items():
        groups_by_shard.setdefault(shard_index, []).append((row_group_index, entries))

    total_groups = len(groups)
    group_index = 0
    for shard_index in sorted(groups_by_shard):
        shard = shards[shard_index]
        parquet_file = pq.ParquetFile(str(shard["url"]))
        shard_groups = sorted(groups_by_shard[shard_index], key=lambda item: item[0])
        for row_group_index, entries in shard_groups:
            group_index += 1
            table = parquet_file.read_row_group(row_group_index, columns=DATA_COLUMNS)
            if on_row_group is not None:
                on_row_group(
                    downloaded_bytes=_estimate_downloaded_bytes(
                        parquet_file,
                        row_group_index,
                        columns=DATA_COLUMNS,
                        fallback_bytes=int(table.nbytes),
                    ),
                    group_index=group_index,
                    total_groups=total_groups,
                )
            sorted_entries = sorted(entries, key=lambda item: item[1])
            taken = table.take(pa.array([offset for _, offset in sorted_entries], type=pa.int64()))
            rows = [freeze_row(row) for row in taken.to_pylist()]
            for i, (row_id, _) in enumerate(sorted_entries):
                fetched[row_id] = rows[i]
    return fetched


def _estimate_downloaded_bytes(
    parquet_file: Any,
    row_group_index: int,
    *,
    columns: list[str],
    fallback_bytes: int,
) -> int:
    metadata = getattr(parquet_file, "metadata", None)
    schema_arrow = getattr(parquet_file, "schema_arrow", None)
    if metadata is None or schema_arrow is None:
        return fallback_bytes
    try:
        row_group_meta = metadata.row_group(row_group_index)
        column_names = list(schema_arrow.names)
        name_to_index = {name: index for index, name in enumerate(column_names)}
        total_bytes = 0
        for column in columns:
            column_index = name_to_index.get(column)
            if column_index is None:
                continue
            total_bytes += int(row_group_meta.column(column_index).total_compressed_size)
        return total_bytes if total_bytes > 0 else fallback_bytes
    except Exception:
        return fallback_bytes


def _fetch_block_entries(
    *,
    catalog: dict[str, Any],
    parquet_files: dict[int, Any],
    row_ids: np.ndarray,
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    groups = _group_row_ids_by_shard(catalog, row_ids)
    if len(groups) != 1:
        raise ValueError("each logical block must map to exactly one row group")
    ((shard_index, row_group_index), entries), = groups.items()
    shards = list(catalog["shards"])
    parquet_file = parquet_files.get(shard_index)
    if parquet_file is None:
        parquet_file = pq.ParquetFile(str(shards[shard_index]["url"]))
        parquet_files[shard_index] = parquet_file

    table = parquet_file.read_row_group(row_group_index, columns=DATA_COLUMNS)
    sorted_entries = sorted(entries, key=lambda item: item[1])
    taken = table.take(pa.array([offset for _, offset in sorted_entries], type=pa.int64()))
    rows = [freeze_row(row) for row in taken.to_pylist()]
    block_entries = [(row_id, rows[i]) for i, (row_id, _) in enumerate(sorted_entries)]
    downloaded_bytes = _estimate_downloaded_bytes(
        parquet_file,
        row_group_index,
        columns=DATA_COLUMNS,
        fallback_bytes=int(table.nbytes),
    )
    return block_entries, downloaded_bytes


def _warm_missing_rows(
    *,
    split: str,
    catalog: dict[str, Any],
    cache_paths,
    cache: SplitRowCache,
    sample_plan: SamplePlan,
    summary: WarmupSummary,
) -> int:
    progress = tqdm(
        total=summary.selected_missing_blocks,
        desc=f"cache {split}",
        unit="blk",
        disable=not is_primary(),
        dynamic_ncols=True,
        leave=False,
        colour="#ff9800",
        smoothing=0.3,
        mininterval=0.3,
    )
    progress_state = WarmupProgressState()
    resolved_blocks = 0

    try:
        _update_warmup_progress(
            progress,
            state=progress_state,
            resolved_blocks=resolved_blocks,
            selected_missing_blocks=summary.selected_missing_blocks,
            selected_block_count=summary.selected_block_count,
        )

        pending_entries: list[tuple[int, dict[str, Any]]] = []
        parquet_files: dict[int, Any] = {}
        chunk_target_rows = CHUNK_TARGET_BLOCKS * BLOCK_SIZE
        for i, _block_index in enumerate(sample_plan.selected_blocks):
            row_ids = _block_row_ids(sample_plan, i)
            if row_ids.size == 0 or np.all(cache.cached[row_ids]):
                continue
            missing_entries, downloaded_bytes = _fetch_block_entries(
                catalog=catalog,
                parquet_files=parquet_files,
                row_ids=row_ids,
            )
            if not missing_entries:
                continue
            pending_entries.extend(missing_entries)
            if len(pending_entries) >= chunk_target_rows:
                materialize_rows(cache_paths, row_entries=pending_entries, cache=cache)
                write_row_cache(cache_paths, cache)
                pending_entries = []
            resolved_blocks += 1
            progress.update(1)
            _update_warmup_progress(
                progress,
                state=progress_state,
                resolved_blocks=resolved_blocks,
                selected_missing_blocks=summary.selected_missing_blocks,
                selected_block_count=summary.selected_block_count,
                downloaded_bytes_delta=downloaded_bytes,
            )
        if pending_entries:
            materialize_rows(cache_paths, row_entries=pending_entries, cache=cache)
            write_row_cache(cache_paths, cache)
    finally:
        _update_warmup_progress(
            progress,
            state=progress_state,
            resolved_blocks=resolved_blocks,
            selected_missing_blocks=summary.selected_missing_blocks,
            selected_block_count=summary.selected_block_count,
            force=True,
        )
        progress.close()
    return resolved_blocks


def ensure_split_cache(
    *,
    split: str,
    dataset_name: str,
    revision: str | None,
    max_samples: int | None,
    seed: int,
    cache_root: Path,
    startup_callback=None,
) -> Path:
    """Ensure that every selected logical block is locally materialized in the row cache."""
    source_root, descriptor = ensure_source_root(
        dataset_name=dataset_name,
        revision=revision,
        cache_root=cache_root,
    )
    catalog = ensure_split_catalog(
        source_root=source_root,
        descriptor=descriptor,
        split=split,
        startup_callback=startup_callback,
    )
    cache_paths = resolve_row_cache_paths(source_root, split)
    sample_plan = plan_sample(catalog, seed, max_samples, split=split)

    with file_lock(cache_paths.lock_path):
        cache = load_or_init_row_cache(cache_paths, total_rows=int(catalog["total_rows"]))
        summary = _build_warmup_summary(sample_plan, cache=cache)
        _emit_warmup_summary(startup_callback, split=split, status="start", summary=summary)

        if summary.cache_only:
            timeline = _render_warmup_timeline(
                sample_plan.selected_bitmap,
                stop_block=sample_plan.execution_block_count,
            )
            _emit_warmup_summary(
                startup_callback,
                split=split,
                status="done",
                summary=summary,
                elapsed_sec=0.0,
                timeline=timeline,
            )
            return resolve_catalog_path(source_root, split)

        started_at = time.perf_counter()
        resolved_blocks = _warm_missing_rows(
            split=split,
            catalog=catalog,
            cache_paths=cache_paths,
            cache=cache,
            sample_plan=sample_plan,
            summary=summary,
        )
        done_summary_fields = summary.event_fields()
        done_summary_fields["resolved_blocks"] = resolved_blocks
        _emit_warmup_summary(
            startup_callback,
            split=split,
            status="done",
            summary=WarmupSummary(**done_summary_fields),
            elapsed_sec=time.perf_counter() - started_at,
            timeline=_render_warmup_timeline(
                sample_plan.selected_bitmap,
                stop_block=sample_plan.execution_block_count,
            ),
        )
    return resolve_catalog_path(source_root, split)


import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_primary() -> bool:
    return get_rank() == 0


__all__ = [
    "WarmupProgressState",
    "WarmupSummary",
    "_compact_warmup_timeline",
    "ensure_split_cache",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_primary",
    "pq",
    "tqdm",
]
