from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .constants import BLOCK_SIZE


SELECTION_PLANNER_MODE = "uniform_exact_k"


@dataclass(slots=True)
class SamplePlan:
    """Logical block sampling result plus the concrete global row ids to materialize."""

    requested_rows: int
    effective_rows: int
    required_blocks: int
    total_blocks: int
    planner_mode: str
    selected_blocks: np.ndarray
    selected_bitmap: np.ndarray
    execution_block_count: int
    selected_row_ids: np.ndarray
    selected_row_offsets: np.ndarray


@dataclass(slots=True)
class SelectionTrace:
    total_blocks: int
    requested_rows: int
    required_blocks: int
    planner_mode: str
    draw_order: np.ndarray
    selected_blocks: np.ndarray
    selected_bitmap: np.ndarray
    execution_block_count: int


def _derive_named_seed(seed: int, split: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{purpose}:{split}".encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "big")


def _select_blocks_uniform_exact_k(
    *,
    required_blocks: int,
    total_blocks: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if required_blocks <= 0 or total_blocks <= 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty
    if required_blocks >= total_blocks:
        selected = np.arange(total_blocks, dtype=np.int64)
        return selected, selected.copy()

    rng = np.random.default_rng(seed)
    draw_order = rng.choice(total_blocks, size=required_blocks, replace=False).astype(np.int64)
    selected_blocks = np.sort(draw_order)
    return selected_blocks, draw_order


def _build_source_blocks(catalog: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    total_rows = int(catalog["total_rows"])
    if total_rows <= 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty

    shards = catalog.get("shards")
    if not isinstance(shards, list):
        starts = np.arange(0, total_rows, BLOCK_SIZE, dtype=np.int64)
        stops = np.minimum(starts + BLOCK_SIZE, total_rows)
        return starts, stops

    starts: list[int] = []
    stops: list[int] = []
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        shard_global_start = int(shard["global_start"])
        current = shard_global_start
        for value in shard.get("row_group_rows", []):
            row_group_rows = int(value)
            if row_group_rows <= 0:
                continue
            row_group_stop = current + row_group_rows
            starts.append(current)
            stops.append(row_group_stop)
            current = row_group_stop
    if starts:
        return np.asarray(starts, dtype=np.int64), np.asarray(stops, dtype=np.int64)
    starts = np.arange(0, total_rows, BLOCK_SIZE, dtype=np.int64)
    stops = np.minimum(starts + BLOCK_SIZE, total_rows)
    return np.asarray(starts, dtype=np.int64), np.asarray(stops, dtype=np.int64)


def _ordered_source_blocks(catalog: dict[str, object], *, seed: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    block_starts, block_stops = _build_source_blocks(catalog)
    if block_starts.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty
    rng = np.random.default_rng(_derive_named_seed(seed, split, "block-order"))
    order = rng.permutation(block_starts.size)
    return block_starts[order], block_stops[order]


def _collect_selected_row_ids(
    *,
    ordered_block_starts: np.ndarray,
    ordered_block_stops: np.ndarray,
    selected_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if selected_blocks.size == 0:
        return np.empty((0,), dtype=np.int64), np.asarray([0], dtype=np.int64)

    offsets = [0]
    segments: list[np.ndarray] = []
    for block_index in selected_blocks:
        start = int(ordered_block_starts[int(block_index)])
        stop = int(ordered_block_stops[int(block_index)])
        segment = np.arange(start, stop, dtype=np.int64)
        segments.append(segment)
        offsets.append(offsets[-1] + int(segment.size))
    return np.concatenate(segments), np.asarray(offsets, dtype=np.int64)

def trace_plan_sample(
    catalog: dict[str, object],
    seed: int,
    max_samples: int | None,
    *,
    split: str = "",
) -> SelectionTrace:
    """Planner trace for examples/debugging."""
    split_seed = _derive_named_seed(seed, split, "selection")
    total_rows = int(catalog["total_rows"])
    block_starts, _ = _build_source_blocks(catalog)
    total_blocks = int(block_starts.size)
    requested_rows = total_rows if max_samples is None else min(max_samples, total_rows)
    if requested_rows <= 0 or total_blocks == 0:
        return SelectionTrace(
            total_blocks=total_blocks,
            requested_rows=0,
            required_blocks=0,
            planner_mode=SELECTION_PLANNER_MODE,
            draw_order=np.empty((0,), dtype=np.int64),
            selected_blocks=np.empty((0,), dtype=np.int64),
            selected_bitmap=np.zeros(total_blocks, dtype=np.bool_),
            execution_block_count=0,
        )

    required_blocks = min(total_blocks, int(math.ceil(requested_rows / BLOCK_SIZE)))
    if required_blocks >= total_blocks:
        selected_blocks = np.arange(total_blocks, dtype=np.int64)
        selected_bitmap = np.zeros(total_blocks, dtype=np.bool_)
        selected_bitmap[selected_blocks] = True
        return SelectionTrace(
            total_blocks=total_blocks,
            requested_rows=requested_rows,
            required_blocks=required_blocks,
            planner_mode=SELECTION_PLANNER_MODE,
            draw_order=selected_blocks.copy(),
            selected_blocks=selected_blocks,
            selected_bitmap=selected_bitmap,
            execution_block_count=(int(selected_blocks[-1]) + 1) if selected_blocks.size else 0,
        )

    selected_blocks, draw_order = _select_blocks_uniform_exact_k(
        required_blocks=required_blocks,
        total_blocks=total_blocks,
        seed=split_seed,
    )
    selected_bitmap = np.zeros(total_blocks, dtype=np.bool_)
    selected_bitmap[selected_blocks] = True
    return SelectionTrace(
        total_blocks=total_blocks,
        requested_rows=requested_rows,
        required_blocks=required_blocks,
        planner_mode=SELECTION_PLANNER_MODE,
        draw_order=draw_order,
        selected_blocks=selected_blocks,
        selected_bitmap=selected_bitmap,
        execution_block_count=int(selected_blocks[-1]) + 1,
    )


def plan_sample(
    catalog: dict[str, object],
    seed: int,
    max_samples: int | None,
    *,
    split: str = "",
) -> SamplePlan:
    """Plan logical blocks and expand them into deterministic global row ids."""
    split_seed = _derive_named_seed(seed, split, "selection")
    total_rows = int(catalog["total_rows"])
    ordered_block_starts, ordered_block_stops = _ordered_source_blocks(catalog, seed=seed, split=split)
    total_blocks = int(ordered_block_starts.size)
    requested_rows = total_rows if max_samples is None else min(max_samples, total_rows)
    if requested_rows <= 0 or total_blocks == 0:
        return SamplePlan(
            requested_rows=0,
            effective_rows=0,
            required_blocks=0,
            total_blocks=total_blocks,
            planner_mode=SELECTION_PLANNER_MODE,
            selected_blocks=np.empty((0,), dtype=np.int64),
            selected_bitmap=np.zeros(total_blocks, dtype=np.bool_),
            execution_block_count=0,
            selected_row_ids=np.empty((0,), dtype=np.int64),
            selected_row_offsets=np.asarray([0], dtype=np.int64),
        )

    required_blocks = min(total_blocks, int(math.ceil(requested_rows / BLOCK_SIZE)))
    selected_blocks, _draw_order = _select_blocks_uniform_exact_k(
        required_blocks=required_blocks,
        total_blocks=total_blocks,
        seed=split_seed,
    )
    selected_bitmap = np.zeros(total_blocks, dtype=np.bool_)
    selected_bitmap[selected_blocks] = True
    selected_row_ids, selected_row_offsets = _collect_selected_row_ids(
        ordered_block_starts=ordered_block_starts,
        ordered_block_stops=ordered_block_stops,
        selected_blocks=selected_blocks,
    )
    return SamplePlan(
        requested_rows=requested_rows,
        effective_rows=int(selected_row_ids.size),
        required_blocks=required_blocks,
        total_blocks=total_blocks,
        planner_mode=SELECTION_PLANNER_MODE,
        selected_blocks=selected_blocks,
        selected_bitmap=selected_bitmap,
        execution_block_count=(int(selected_blocks[-1]) + 1) if selected_blocks.size else 0,
        selected_row_ids=selected_row_ids,
        selected_row_offsets=selected_row_offsets,
    )

__all__ = [
    "SELECTION_PLANNER_MODE",
    "SamplePlan",
    "SelectionTrace",
    "plan_sample",
    "trace_plan_sample",
]
