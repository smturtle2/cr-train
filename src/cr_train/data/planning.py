from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .constants import BLOCK_SIZE


SELECTION_PLANNER_MODE = "uniform_exact_k"
FULL_SPLIT_PLANNER_MODE = "full_split"


@dataclass(slots=True)
class SamplePlan:
    """Logical block sampling result."""

    requested_rows: int
    effective_rows: int
    required_blocks: int
    total_blocks: int
    planner_mode: str
    selected_blocks: np.ndarray
    selected_bitmap: np.ndarray
    execution_block_count: int


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


def _select_all_blocks(total_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    selected_blocks = np.arange(total_blocks, dtype=np.int64)
    selected_bitmap = np.ones(total_blocks, dtype=np.bool_)
    return selected_blocks, selected_bitmap


def _resolve_total_blocks(catalog: dict[str, object]) -> int:
    block_row_counts = catalog.get("block_row_counts")
    if isinstance(block_row_counts, list):
        return len(block_row_counts)
    if "total_blocks" in catalog:
        return max(0, int(catalog["total_blocks"]))

    total_rows = int(catalog.get("total_rows", 0))
    if total_rows <= 0:
        return 0
    return int(math.ceil(total_rows / BLOCK_SIZE))


def _estimate_effective_rows(
    catalog: dict[str, object],
    *,
    requested_rows: int,
    selected_blocks: np.ndarray,
    total_blocks: int,
) -> int:
    if selected_blocks.size == 0:
        return 0

    block_row_counts = catalog.get("block_row_counts")
    if isinstance(block_row_counts, list):
        return int(sum(int(block_row_counts[int(index)]) for index in selected_blocks.tolist()))

    total_rows = int(catalog.get("total_rows", 0))
    if total_rows <= 0:
        return int(selected_blocks.size) * BLOCK_SIZE
    if selected_blocks.size >= total_blocks:
        return total_rows
    return min(total_rows, int(selected_blocks.size) * BLOCK_SIZE)


def _resolve_requested_rows(total_rows: int, max_samples: int | None) -> int:
    return total_rows if max_samples is None else min(max_samples, total_rows)


def _is_full_split_request(*, total_rows: int, requested_rows: int) -> bool:
    return total_rows > 0 and requested_rows >= total_rows


def trace_plan_sample(
    catalog: dict[str, object],
    seed: int,
    max_samples: int | None,
    *,
    split: str = "",
) -> SelectionTrace:
    """Planner trace for examples/debugging."""
    split_seed = _derive_named_seed(seed, split, "selection")
    total_rows = int(catalog.get("total_rows", 0))
    total_blocks = _resolve_total_blocks(catalog)
    requested_rows = _resolve_requested_rows(total_rows, max_samples)
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

    if _is_full_split_request(total_rows=total_rows, requested_rows=requested_rows):
        selected_blocks, selected_bitmap = _select_all_blocks(total_blocks)
        return SelectionTrace(
            total_blocks=total_blocks,
            requested_rows=requested_rows,
            required_blocks=total_blocks,
            planner_mode=FULL_SPLIT_PLANNER_MODE,
            draw_order=selected_blocks.copy(),
            selected_blocks=selected_blocks,
            selected_bitmap=selected_bitmap,
            execution_block_count=total_blocks,
        )

    required_blocks = min(total_blocks, int(math.ceil(requested_rows / BLOCK_SIZE)))
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
    """Plan logical blocks without binding to a row-indexed cache layout."""
    split_seed = _derive_named_seed(seed, split, "selection")
    total_rows = int(catalog.get("total_rows", 0))
    total_blocks = _resolve_total_blocks(catalog)
    requested_rows = _resolve_requested_rows(total_rows, max_samples)
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
        )

    if _is_full_split_request(total_rows=total_rows, requested_rows=requested_rows):
        selected_blocks, selected_bitmap = _select_all_blocks(total_blocks)
        effective_rows = _estimate_effective_rows(
            catalog,
            requested_rows=requested_rows,
            selected_blocks=selected_blocks,
            total_blocks=total_blocks,
        )
        return SamplePlan(
            requested_rows=requested_rows,
            effective_rows=effective_rows,
            required_blocks=total_blocks,
            total_blocks=total_blocks,
            planner_mode=FULL_SPLIT_PLANNER_MODE,
            selected_blocks=selected_blocks,
            selected_bitmap=selected_bitmap,
            execution_block_count=total_blocks,
        )

    required_blocks = min(total_blocks, int(math.ceil(requested_rows / BLOCK_SIZE)))
    selected_blocks, _draw_order = _select_blocks_uniform_exact_k(
        required_blocks=required_blocks,
        total_blocks=total_blocks,
        seed=split_seed,
    )
    selected_bitmap = np.zeros(total_blocks, dtype=np.bool_)
    selected_bitmap[selected_blocks] = True
    effective_rows = _estimate_effective_rows(
        catalog,
        requested_rows=requested_rows,
        selected_blocks=selected_blocks,
        total_blocks=total_blocks,
    )
    return SamplePlan(
        requested_rows=requested_rows,
        effective_rows=effective_rows,
        required_blocks=required_blocks,
        total_blocks=total_blocks,
        planner_mode=SELECTION_PLANNER_MODE,
        selected_blocks=selected_blocks,
        selected_bitmap=selected_bitmap,
        execution_block_count=(int(selected_blocks[-1]) + 1) if selected_blocks.size else 0,
    )


__all__ = [
    "SELECTION_PLANNER_MODE",
    "SamplePlan",
    "SelectionTrace",
    "plan_sample",
    "trace_plan_sample",
]
