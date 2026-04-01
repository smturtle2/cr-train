from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm

from .constants import (
    BLOCK_SIZE,
    CANONICAL_SHUFFLE_BUFFER_SIZE,
    DATA_COLUMNS,
    WARMUP_SPEED_EMA_ALPHA,
    WARMUP_TIMELINE_WIDTH,
)
from .planning import CachePlan, build_cache_plan, plan_sample
from .source import emit_startup_event, ensure_source_root, ensure_split_catalog, resolve_catalog_path
from .store import (
    SplitBlockCache,
    freeze_row,
    load_or_init_block_cache,
    materialize_blocks,
    resolve_block_cache_paths,
    resolve_dataset_seed,
    suppress_hf_datasets_progress_bars,
    write_block_cache,
)


@dataclass(slots=True)
class WarmupProgressState:
    ema_blocks_per_sec: float | None = None
    last_update_at: float | None = None
    last_resolved_blocks: int = 0


@dataclass(slots=True)
class WarmupSummary:
    dataset_seed: int
    requested_rows: int
    effective_rows: int
    required_blocks: int
    candidate_blocks: int
    planner_mode: str
    base_take_prob: float
    cached_blocks: int
    missing_blocks: int
    cache_only: bool
    run_count: int
    compressed_block_count: int
    frontier_run_count: int
    resolved_blocks: int = 0

    @classmethod
    def from_cache_plan(cls, dataset_seed: int, cache_plan: CachePlan, *, resolved_blocks: int = 0) -> "WarmupSummary":
        sample_plan = cache_plan.sample_plan
        return cls(
            dataset_seed=dataset_seed,
            requested_rows=sample_plan.requested_rows,
            effective_rows=sample_plan.effective_rows,
            required_blocks=sample_plan.required_blocks,
            candidate_blocks=sample_plan.candidate_blocks,
            planner_mode=sample_plan.planner_mode,
            base_take_prob=sample_plan.base_take_prob,
            cached_blocks=cache_plan.cached_blocks,
            missing_blocks=cache_plan.missing_blocks,
            cache_only=cache_plan.cache_only,
            run_count=len(cache_plan.execution_runs),
            compressed_block_count=sample_plan.execution_block_count,
            frontier_run_count=len(cache_plan.frontier_runs),
            resolved_blocks=resolved_blocks,
        )

    def event_fields(self) -> dict[str, Any]:
        return {
            "dataset_seed": self.dataset_seed,
            "requested_rows": self.requested_rows,
            "effective_rows": self.effective_rows,
            "required_blocks": self.required_blocks,
            "candidate_blocks": self.candidate_blocks,
            "planner_mode": self.planner_mode,
            "base_take_prob": self.base_take_prob,
            "cached_blocks": self.cached_blocks,
            "missing_blocks": self.missing_blocks,
            "cache_only": self.cache_only,
            "run_count": self.run_count,
            "compressed_block_count": self.compressed_block_count,
            "frontier_run_count": self.frontier_run_count,
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


def _iter_canonical_stream_rows(
    *,
    split: str,
    dataset_name: str,
    revision: str | None,
    dataset_seed: int,
    cache_root: Path,
):
    with suppress_hf_datasets_progress_bars():
        dataset = load_dataset(
            dataset_name,
            split=split,
            revision=revision,
            streaming=True,
            cache_dir=str(cache_root),
        )
        if hasattr(dataset, "select_columns"):
            dataset = dataset.select_columns(DATA_COLUMNS)
        dataset = dataset.shuffle(seed=dataset_seed, buffer_size=CANONICAL_SHUFFLE_BUFFER_SIZE)
        for row in dataset:
            yield dict(row)


def _skip_canonical_blocks(row_iterator, block_count: int) -> None:
    for _ in range(block_count):
        saw_row = False
        for _ in range(BLOCK_SIZE):
            try:
                next(row_iterator)
                saw_row = True
            except StopIteration:
                if not saw_row:
                    raise ValueError("canonical stream ended before the expected frontier")
                break


def _read_canonical_blocks(row_iterator, block_count: int) -> list[list[dict[str, Any]]]:
    blocks: list[list[dict[str, Any]]] = []
    for _ in range(block_count):
        block_rows: list[dict[str, Any]] = []
        for _ in range(BLOCK_SIZE):
            try:
                block_rows.append(freeze_row(next(row_iterator)))
            except StopIteration:
                break
        if not block_rows:
            raise ValueError("canonical stream ended before the expected block range")
        blocks.append(block_rows)
    return blocks


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


def _update_warmup_progress(
    progress: Any,
    *,
    state: WarmupProgressState,
    resolved_blocks: int,
    missing_blocks: int,
    cached_blocks: int,
    force: bool = False,
) -> None:
    if getattr(progress, "disable", False):
        return

    now = time.perf_counter()
    if state.last_update_at is None:
        state.last_update_at = now
        state.last_resolved_blocks = resolved_blocks
    else:
        delta_blocks = resolved_blocks - state.last_resolved_blocks
        elapsed = max(now - state.last_update_at, 1e-9)
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
            state.last_update_at = now
        elif force and state.ema_blocks_per_sec is None:
            state.ema_blocks_per_sec = 0.0

    progress.set_postfix(
        {
            "miss": f"{resolved_blocks}/{missing_blocks}",
            "hit": cached_blocks,
            "blk/s": f"{(state.ema_blocks_per_sec or 0.0):.1f}",
        }
    )


def _warm_missing_blocks(
    *,
    split: str,
    dataset_name: str,
    revision: str | None,
    dataset_seed: int,
    cache_root: Path,
    cache_paths,
    cache: SplitBlockCache,
    cache_plan: CachePlan,
) -> int:
    total_frontier_blocks = cache_plan.stream_start_block + sum(
        run.block_count for run in cache_plan.frontier_runs
    )
    resolved_blocks = 0
    progress = tqdm(
        total=total_frontier_blocks,
        desc=f"cache {split}",
        unit="blk",
        disable=not is_primary(),
        dynamic_ncols=True,
        leave=False,
    )
    progress_state = WarmupProgressState()
    row_iterator = _iter_canonical_stream_rows(
        split=split,
        dataset_name=dataset_name,
        revision=revision,
        dataset_seed=dataset_seed,
        cache_root=cache_root,
    )
    def _desc(label: str) -> None:
        if hasattr(progress, "set_description_str"):
            progress.set_description_str(f"cache {split} {label}", refresh=False)

    try:
        if cache_plan.stream_start_block > 0:
            _desc("seek")
            for _ in range(cache_plan.stream_start_block):
                _skip_canonical_blocks(row_iterator, 1)
                progress.update(1)
        for run in cache_plan.frontier_runs:
            if run.kind == "skip":
                _desc(f"skip {run.block_count}blk")
                for _ in range(run.block_count):
                    _skip_canonical_blocks(row_iterator, 1)
                    progress.update(1)
                cache.state.canonical_frontier_block = max(cache.state.canonical_frontier_block, run.stop_block)
                write_block_cache(cache_paths, cache)
                continue

            _desc(f"fetch blk {run.start_block}-{run.stop_block - 1}")
            resolved_blocks += materialize_blocks(
                cache_paths,
                start_block=run.start_block,
                blocks=_read_canonical_blocks(row_iterator, run.block_count),
                cache=cache,
            )
            cache.state.canonical_frontier_block = max(cache.state.canonical_frontier_block, run.stop_block)
            write_block_cache(cache_paths, cache)
            progress.update(run.block_count)
            _update_warmup_progress(
                progress,
                state=progress_state,
                resolved_blocks=resolved_blocks,
                missing_blocks=cache_plan.missing_blocks,
                cached_blocks=cache_plan.cached_blocks,
            )
    finally:
        row_iterator.close()
        _update_warmup_progress(
            progress,
            state=progress_state,
            resolved_blocks=resolved_blocks,
            missing_blocks=cache_plan.missing_blocks,
            cached_blocks=cache_plan.cached_blocks,
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
    dataset_seed: int | None,
    cache_root: Path,
    startup_callback=None,
) -> Path:
    source_root, descriptor = ensure_source_root(
        dataset_name=dataset_name,
        revision=revision,
        cache_root=cache_root,
    )
    resolved_dataset_seed = resolve_dataset_seed(dataset_seed)
    cache_paths = resolve_block_cache_paths(
        source_root,
        split,
        resolved_dataset_seed,
        CANONICAL_SHUFFLE_BUFFER_SIZE,
    )
    catalog = ensure_split_catalog(
        source_root=source_root,
        descriptor=descriptor,
        split=split,
        startup_callback=startup_callback,
    )
    sample_plan = plan_sample(catalog, seed, max_samples, split=split)
    cache = load_or_init_block_cache(
        cache_paths,
        dataset_seed=resolved_dataset_seed,
        shuffle_buffer_size=CANONICAL_SHUFFLE_BUFFER_SIZE,
        total_rows=int(catalog["total_rows"]),
    )
    cache_plan = build_cache_plan(
        sample_plan,
        cache.cached,
        frontier_block=cache.state.canonical_frontier_block,
    )

    summary = WarmupSummary.from_cache_plan(resolved_dataset_seed, cache_plan)
    _emit_warmup_summary(startup_callback, split=split, status="start", summary=summary)

    if cache_plan.cache_only:
        timeline = _render_warmup_timeline(
            cache_plan.sample_plan.selected_bitmap,
            stop_block=cache_plan.sample_plan.execution_block_count,
        )
        _emit_warmup_summary(
            startup_callback, split=split, status="done",
            summary=summary, elapsed_sec=0.0, timeline=timeline,
        )
        return resolve_catalog_path(source_root, split)

    started_at = time.perf_counter()
    resolved_blocks = _warm_missing_blocks(
        split=split,
        dataset_name=dataset_name,
        revision=revision,
        dataset_seed=resolved_dataset_seed,
        cache_root=cache_root,
        cache_paths=cache_paths,
        cache=cache,
        cache_plan=cache_plan,
    )
    _emit_warmup_summary(
        startup_callback, split=split, status="done",
        summary=WarmupSummary.from_cache_plan(
            resolved_dataset_seed, cache_plan, resolved_blocks=resolved_blocks,
        ),
        elapsed_sec=time.perf_counter() - started_at,
        timeline=_render_warmup_timeline(
            cache_plan.sample_plan.selected_bitmap,
            stop_block=cache_plan.sample_plan.execution_block_count,
        ),
    )
    return resolve_catalog_path(source_root, split)


try:
    import torch.distributed as dist
except Exception:  # pragma: no cover - torch without distributed support
    dist = None


def is_distributed() -> bool:
    return bool(dist and dist.is_available() and dist.is_initialized())


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_primary() -> bool:
    return get_rank() == 0


__all__ = [
    "WarmupProgressState",
    "WarmupSummary",
    "ensure_split_cache",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_primary",
    "load_dataset",
    "tqdm",
]
