from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .constants import BLOCK_SIZE, STOP_BIAS_ALPHA, STOP_BIAS_MIX


SELECTION_PLANNER_MODE = "stop_biased_exact_k"


@dataclass(slots=True)
class SamplePlan:
    """Logical block sampling result plus the concrete global row ids to materialize."""

    requested_rows: int
    effective_rows: int
    required_blocks: int
    total_blocks: int
    planner_mode: str
    stop_bias_alpha: float
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
    stop_bias_alpha: float
    stop_candidates: np.ndarray
    stop_probabilities: np.ndarray
    sampled_stop_block: int | None
    prefix_draws: np.ndarray
    selected_blocks: np.ndarray
    selected_bitmap: np.ndarray
    execution_block_count: int


def compute_stop_probability(
    required_blocks: int,
    total_blocks: int,
    stop_block: int,
    *,
    stop_bias_alpha: float = STOP_BIAS_ALPHA,
) -> float:
    """Probability mass over stop blocks across the full logical block order."""
    if required_blocks <= 0 or total_blocks <= 0:
        return 0.0
    min_stop = required_blocks - 1
    max_stop = total_blocks - 1
    if stop_block < min_stop or stop_block > max_stop:
        return 0.0
    positions = np.arange(min_stop, max_stop + 1, dtype=np.float64)
    denom = max(total_blocks - required_blocks, 1)
    weights = np.exp(-stop_bias_alpha * ((positions - min_stop) / denom))
    weights /= float(np.sum(weights))
    uniform = np.full_like(weights, 1.0 / len(weights))
    mixed_weights = (STOP_BIAS_MIX * weights) + ((1.0 - STOP_BIAS_MIX) * uniform)
    weights_sum = float(np.sum(mixed_weights))
    if weights_sum <= 0.0:
        return 0.0
    return float(mixed_weights[int(stop_block - min_stop)] / weights_sum)


def _derive_named_seed(seed: int, split: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{purpose}:{split}".encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "big")


def _sample_stop_block(
    *,
    required_blocks: int,
    total_blocks: int,
    rng: np.random.Generator,
    stop_bias_alpha: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    min_stop = required_blocks - 1
    max_stop = total_blocks - 1
    if min_stop >= max_stop:
        stop_candidates = np.asarray([min_stop], dtype=np.int64)
        probabilities = np.asarray([1.0], dtype=np.float64)
        return min_stop, stop_candidates, probabilities
    stop_candidates = np.arange(min_stop, max_stop + 1, dtype=np.int64)
    probabilities = np.asarray(
        [
            compute_stop_probability(
                required_blocks,
                total_blocks,
                int(stop_block),
                stop_bias_alpha=stop_bias_alpha,
            )
            for stop_block in stop_candidates
        ],
        dtype=np.float64,
    )
    return int(rng.choice(stop_candidates, p=probabilities)), stop_candidates, probabilities


def _select_blocks_stop_biased(
    *,
    required_blocks: int,
    total_blocks: int,
    seed: int,
    stop_bias_alpha: float,
) -> tuple[np.ndarray, float]:
    if required_blocks <= 0 or total_blocks <= 0:
        return np.empty((0,), dtype=np.int64), stop_bias_alpha
    if required_blocks >= total_blocks:
        return np.arange(total_blocks, dtype=np.int64), stop_bias_alpha

    rng = np.random.default_rng(seed)
    stop_block, _, _ = _sample_stop_block(
        required_blocks=required_blocks,
        total_blocks=total_blocks,
        rng=rng,
        stop_bias_alpha=stop_bias_alpha,
    )
    if required_blocks == 1:
        return np.asarray([stop_block], dtype=np.int64), stop_bias_alpha
    prefix_blocks = rng.choice(stop_block, size=required_blocks - 1, replace=False).astype(np.int64)
    selected_blocks = np.sort(np.concatenate((prefix_blocks, np.asarray([stop_block], dtype=np.int64))))
    return selected_blocks, stop_bias_alpha


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
            stop_bias_alpha=STOP_BIAS_ALPHA,
            stop_candidates=np.empty((0,), dtype=np.int64),
            stop_probabilities=np.empty((0,), dtype=np.float64),
            sampled_stop_block=None,
            prefix_draws=np.empty((0,), dtype=np.int64),
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
            stop_bias_alpha=STOP_BIAS_ALPHA,
            stop_candidates=np.asarray([total_blocks - 1], dtype=np.int64),
            stop_probabilities=np.asarray([1.0], dtype=np.float64),
            sampled_stop_block=(total_blocks - 1) if total_blocks > 0 else None,
            prefix_draws=selected_blocks[:-1],
            selected_blocks=selected_blocks,
            selected_bitmap=selected_bitmap,
            execution_block_count=(int(selected_blocks[-1]) + 1) if selected_blocks.size else 0,
        )

    rng = np.random.default_rng(split_seed)
    sampled_stop_block, stop_candidates, stop_probabilities = _sample_stop_block(
        required_blocks=required_blocks,
        total_blocks=total_blocks,
        rng=rng,
        stop_bias_alpha=STOP_BIAS_ALPHA,
    )
    if required_blocks == 1:
        prefix_draws = np.empty((0,), dtype=np.int64)
        selected_blocks = np.asarray([sampled_stop_block], dtype=np.int64)
    else:
        prefix_draws = np.sort(rng.choice(sampled_stop_block, size=required_blocks - 1, replace=False).astype(np.int64))
        selected_blocks = np.concatenate((prefix_draws, np.asarray([sampled_stop_block], dtype=np.int64)))

    selected_bitmap = np.zeros(total_blocks, dtype=np.bool_)
    selected_bitmap[selected_blocks] = True
    return SelectionTrace(
        total_blocks=total_blocks,
        requested_rows=requested_rows,
        required_blocks=required_blocks,
        planner_mode=SELECTION_PLANNER_MODE,
        stop_bias_alpha=STOP_BIAS_ALPHA,
        stop_candidates=stop_candidates,
        stop_probabilities=stop_probabilities,
        sampled_stop_block=sampled_stop_block,
        prefix_draws=prefix_draws,
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
            stop_bias_alpha=STOP_BIAS_ALPHA,
            selected_blocks=np.empty((0,), dtype=np.int64),
            selected_bitmap=np.zeros(total_blocks, dtype=np.bool_),
            execution_block_count=0,
            selected_row_ids=np.empty((0,), dtype=np.int64),
            selected_row_offsets=np.asarray([0], dtype=np.int64),
        )

    required_blocks = min(total_blocks, int(math.ceil(requested_rows / BLOCK_SIZE)))
    selected_blocks, stop_bias_alpha = _select_blocks_stop_biased(
        required_blocks=required_blocks,
        total_blocks=total_blocks,
        seed=split_seed,
        stop_bias_alpha=STOP_BIAS_ALPHA,
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
        stop_bias_alpha=stop_bias_alpha,
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
    "compute_stop_probability",
    "plan_sample",
    "trace_plan_sample",
]
