from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .constants import BLOCK_SIZE, WARMUP_DOWNLOAD_SPEED_WINDOW_SEC, WARMUP_SPEED_EMA_ALPHA, WARMUP_TIMELINE_WIDTH
from .planning import SamplePlan, plan_sample
from .source import emit_startup_event, ensure_source_root, ensure_split_catalog, load_block_rows, resolve_catalog_path
from ..progress import resolve_progress_bar_ncols, set_progress_postfix_str
from .store import (
    BlockCachePaths,
    block_lock_path,
    clear_block_cache_entry,
    find_completed_block_row_count,
    file_lock,
    load_completed_block_index,
    load_block_metadata,
    resolve_block_cache_paths,
    save_block,
    write_completed_block_marker,
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
            "selected_block_count": self.selected_block_count,
            "cached_selected_blocks": self.cached_selected_blocks,
            "selected_missing_blocks": self.selected_missing_blocks,
            "cache_only": self.cache_only,
            "execution_block_count": self.execution_block_count,
            "resolved_blocks": self.resolved_blocks,
        }


@dataclass(slots=True)
class BlockCacheFillResult:
    cache_key: str
    status: str
    row_count: int
    downloaded_bytes: int
    written_bytes: int
    elapsed_sec: float


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

    if state.ema_download_bytes_per_sec is not None:
        display_download_bytes_per_sec = state.ema_download_bytes_per_sec
    elif state.pending_download_bytes > 0 and state.download_window_started_at is not None:
        elapsed = max(now - state.download_window_started_at, 1e-9)
        display_download_bytes_per_sec = state.pending_download_bytes / elapsed
    else:
        display_download_bytes_per_sec = 0.0
    download_speed_mb_per_sec = display_download_bytes_per_sec / (1024.0 * 1024.0)
    postfix_parts = [f"sel: {selected_block_count}"]
    postfix_parts.append(_format_rate(download_speed_mb_per_sec, "MB/s"))
    set_progress_postfix_str(
        progress,
        ", ".join(postfix_parts),
    )


def _selected_blocks(catalog: dict[str, Any], sample_plan: SamplePlan) -> list[dict[str, Any]]:
    blocks = list(catalog.get("blocks", []))
    return [blocks[int(index)] for index in sample_plan.selected_blocks.tolist()]


def _normalize_row_groups(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    return tuple(int(item) for item in value)


def _metadata_row_count_if_matching(metadata: dict[str, Any] | None, block: dict[str, Any]) -> int | None:
    if metadata is None:
        return None

    row_count = int(metadata.get("row_count", 0))
    if row_count <= 0 or row_count > BLOCK_SIZE:
        return None
    if "shard_index" not in metadata:
        return None
    if int(metadata["shard_index"]) != int(block["shard_index"]):
        return None
    if str(metadata.get("source_file")) != str(block["source_file"]):
        return None

    metadata_row_groups = _normalize_row_groups(metadata.get("row_groups"))
    block_row_groups = tuple(int(value) for value in block["row_groups"])
    if metadata_row_groups != block_row_groups:
        return None
    return row_count


def _load_matching_block_row_count(
    cache_paths: BlockCachePaths,
    block: dict[str, Any],
) -> int | None:
    return _metadata_row_count_if_matching(
        load_block_metadata(cache_paths, str(block["cache_key"])),
        block,
    )


def _resolve_effective_rows(
    *,
    selected_blocks: list[dict[str, Any]],
    completed_by_key: dict[str, int],
    fallback_rows: int,
) -> int:
    total_rows = 0
    for block in selected_blocks:
        row_count = completed_by_key.get(str(block["cache_key"]))
        if row_count is None:
            return fallback_rows
        total_rows += row_count
    return total_rows


def _build_warmup_summary(
    sample_plan: SamplePlan,
    *,
    catalog: dict[str, Any],
    completed_by_key: dict[str, int],
) -> WarmupSummary:
    selected_blocks = _selected_blocks(catalog, sample_plan)
    selected_missing_blocks = sum(
        1 for block in selected_blocks if str(block["cache_key"]) not in completed_by_key
    )
    cached_selected_blocks = int(sample_plan.required_blocks - selected_missing_blocks)
    execution_block_count = int(sample_plan.execution_block_count)
    effective_rows = _resolve_effective_rows(
        selected_blocks=selected_blocks,
        completed_by_key=completed_by_key,
        fallback_rows=sample_plan.effective_rows,
    )
    return WarmupSummary(
        requested_rows=sample_plan.requested_rows,
        effective_rows=effective_rows,
        required_blocks=sample_plan.required_blocks,
        planner_mode=sample_plan.planner_mode,
        selected_block_count=sample_plan.required_blocks,
        cached_selected_blocks=cached_selected_blocks,
        selected_missing_blocks=selected_missing_blocks,
        cache_only=selected_missing_blocks == 0,
        execution_block_count=execution_block_count,
    )


def _ensure_block_cached(
    *,
    dataset_name: str,
    revision: str | None,
    split: str,
    cache_paths: BlockCachePaths,
    block: dict[str, Any],
    startup_callback=None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BlockCacheFillResult:
    cache_key = str(block["cache_key"])
    completed_row_count = find_completed_block_row_count(cache_paths, cache_key)
    if completed_row_count is not None:
        return BlockCacheFillResult(
            cache_key=cache_key,
            status="hit",
            row_count=completed_row_count,
            downloaded_bytes=0,
            written_bytes=0,
            elapsed_sec=0.0,
        )

    started_at = time.perf_counter()
    with file_lock(block_lock_path(cache_paths, cache_key)):
        completed_row_count = find_completed_block_row_count(cache_paths, cache_key)
        if completed_row_count is not None:
            return BlockCacheFillResult(
                cache_key=cache_key,
                status="hit_after_lock",
                row_count=completed_row_count,
                downloaded_bytes=0,
                written_bytes=0,
                elapsed_sec=time.perf_counter() - started_at,
            )
        legacy_row_count = _load_matching_block_row_count(cache_paths, block)
        if legacy_row_count is not None:
            write_completed_block_marker(cache_paths, cache_key, row_count=legacy_row_count)
            return BlockCacheFillResult(
                cache_key=cache_key,
                status="legacy_hit",
                row_count=legacy_row_count,
                downloaded_bytes=0,
                written_bytes=0,
                elapsed_sec=time.perf_counter() - started_at,
            )
        clear_block_cache_entry(cache_paths, cache_key, keep_lock=True)
        downloaded_bytes = 0

        def _on_row_loaded(row_count: int, downloaded_bytes_delta: int) -> None:
            nonlocal downloaded_bytes
            downloaded_bytes += downloaded_bytes_delta
            if progress_callback is not None:
                progress_callback(row_count, downloaded_bytes_delta)

        rows = load_block_rows(
            dataset_name=dataset_name,
            revision=revision,
            split=split,
            block=block,
            progress_callback=_on_row_loaded,
            startup_callback=startup_callback,
        )
        save_result = save_block(
            cache_paths,
            cache_key=cache_key,
            rows=rows,
            metadata={
                "cache_key": cache_key,
                "split": split,
                "block_index": int(block["index"]),
                "shard_index": int(block["shard_index"]),
                "source_file": str(block["source_file"]),
                "row_groups": list(block["row_groups"]),
                "row_count": len(rows),
            },
        )
        return BlockCacheFillResult(
            cache_key=cache_key,
            status="filled",
            row_count=len(rows),
            downloaded_bytes=downloaded_bytes,
            written_bytes=save_result.written_bytes,
            elapsed_sec=time.perf_counter() - started_at,
        )


def _warm_missing_blocks(
    *,
    split: str,
    dataset_name: str,
    revision: str | None,
    catalog: dict[str, Any],
    cache_paths: BlockCachePaths,
    sample_plan: SamplePlan,
    summary: WarmupSummary,
    completed_by_key: dict[str, int],
    startup_callback=None,
) -> int:
    progress = tqdm(
        total=summary.selected_missing_blocks,
        desc=f"cache {split}",
        unit="blk",
        disable=not is_primary(),
        ncols=resolve_progress_bar_ncols(),
        leave=False,
        colour="#ff9800",
        smoothing=0.3,
        mininterval=0.3,
    )
    progress_state = WarmupProgressState()
    resolved_blocks = 0
    selected_blocks = _selected_blocks(catalog, sample_plan)
    missing_blocks = [
        block for block in selected_blocks if str(block["cache_key"]) not in completed_by_key
    ]

    try:
        _update_warmup_progress(
            progress,
            state=progress_state,
            resolved_blocks=resolved_blocks,
            selected_missing_blocks=summary.selected_missing_blocks,
            selected_block_count=summary.selected_block_count,
        )
        for block in missing_blocks:
            result = _ensure_block_cached(
                dataset_name=dataset_name,
                revision=revision,
                split=split,
                cache_paths=cache_paths,
                block=block,
                startup_callback=startup_callback,
                progress_callback=lambda _row_count, downloaded_bytes_delta: _update_warmup_progress(
                    progress,
                    state=progress_state,
                    resolved_blocks=resolved_blocks,
                    selected_missing_blocks=summary.selected_missing_blocks,
                    selected_block_count=summary.selected_block_count,
                    downloaded_bytes_delta=downloaded_bytes_delta,
                ),
            )
            resolved_blocks += 1
            progress.update(1)
            _update_warmup_progress(
                progress,
                state=progress_state,
                resolved_blocks=resolved_blocks,
                selected_missing_blocks=summary.selected_missing_blocks,
                selected_block_count=summary.selected_block_count,
                downloaded_bytes_delta=0,
            )
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
    """Ensure that every selected logical block is locally materialized in the block cache."""
    source_root, descriptor = ensure_source_root(
        dataset_name=dataset_name,
        revision=revision,
        cache_root=cache_root,
        startup_callback=startup_callback,
    )
    catalog = ensure_split_catalog(
        source_root=source_root,
        descriptor=descriptor,
        split=split,
        startup_callback=startup_callback,
    )
    cache_paths = resolve_block_cache_paths(source_root, split)
    sample_plan = plan_sample(catalog, seed, max_samples, split=split)
    completed_by_key = load_completed_block_index(cache_paths)
    summary = _build_warmup_summary(sample_plan, catalog=catalog, completed_by_key=completed_by_key)
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
    resolved_blocks = _warm_missing_blocks(
        split=split,
        dataset_name=dataset_name,
        revision=revision,
        catalog=catalog,
        cache_paths=cache_paths,
        sample_plan=sample_plan,
        summary=summary,
        completed_by_key=completed_by_key,
        startup_callback=startup_callback,
    )
    completed_by_key = load_completed_block_index(cache_paths)
    actual_effective_rows = _resolve_effective_rows(
        selected_blocks=_selected_blocks(catalog, sample_plan),
        completed_by_key=completed_by_key,
        fallback_rows=sample_plan.effective_rows,
    )
    done_summary = WarmupSummary(
        requested_rows=summary.requested_rows,
        effective_rows=actual_effective_rows,
        required_blocks=summary.required_blocks,
        planner_mode=summary.planner_mode,
        selected_block_count=summary.selected_block_count,
        cached_selected_blocks=summary.cached_selected_blocks,
        selected_missing_blocks=summary.selected_missing_blocks,
        cache_only=False,
        execution_block_count=summary.execution_block_count,
        resolved_blocks=resolved_blocks,
    )
    _emit_warmup_summary(
        startup_callback,
        split=split,
        status="done",
        summary=done_summary,
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
    "tqdm",
]
