from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import BLOCK_SIZE, CANDIDATE_WINDOW_FACTOR


SELECTION_PLANNER_MODE = "sequential_additive_exact_k"


@dataclass(slots=True)
class SamplePlan:
    """블록 단위 샘플 선택 결과. selected_bitmap이 어떤 블록을 사용할지 결정."""

    requested_rows: int
    effective_rows: int
    required_blocks: int
    total_blocks: int
    candidate_blocks: int
    planner_mode: str
    base_take_prob: float
    selected_blocks: np.ndarray
    selected_bitmap: np.ndarray
    execution_block_count: int


@dataclass(slots=True)
class ExecutionRun:
    """동일한 실행 종류(skip/take_cached/take_remote/materialize)의 연속 블록 범위."""

    kind: str
    start_block: int
    stop_block: int
    block_count: int
    total_rows: int


@dataclass(slots=True)
class CachePlan:
    """sample plan과 현재 캐시 상태를 결합한 warmup 실행 계획."""

    sample_plan: SamplePlan
    hit_bitmap: np.ndarray
    missing_bitmap: np.ndarray
    cached_blocks: int
    missing_blocks: int
    cache_only: bool
    execution_runs: list[ExecutionRun]
    frontier_runs: list[ExecutionRun]
    stream_start_block: int


def compute_base_take_probability(required_blocks: int, candidate_blocks: int) -> float:
    """블록 선택 기본 확률: required / candidate."""
    if required_blocks <= 0 or candidate_blocks <= 0:
        return 0.0
    return min(1.0, required_blocks / candidate_blocks)


def compute_take_probability(
    required_blocks: int,
    candidate_blocks: int,
    remaining_candidates: int,
) -> float:
    """후보 소진에 따라 증가하는 동적 선택 확률. 정확한 블록 수 보장을 위한 보정 포함."""
    if required_blocks <= 0 or candidate_blocks <= 0 or remaining_candidates <= 0:
        return 0.0
    base_take_prob = compute_base_take_probability(required_blocks, candidate_blocks)
    dynamic_bonus = max(0.0, (required_blocks / remaining_candidates) - base_take_prob)
    return min(1.0, base_take_prob + dynamic_bonus)


def _select_blocks_sequentially(
    *,
    required_blocks: int,
    candidate_blocks: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    base_take_prob = compute_base_take_probability(required_blocks, candidate_blocks)
    if required_blocks <= 0 or candidate_blocks <= 0:
        return np.empty((0,), dtype=np.int64), base_take_prob
    if required_blocks >= candidate_blocks:
        return np.arange(candidate_blocks, dtype=np.int64), base_take_prob

    # split별 독립 시드로 블록 선택 재현성 보장
    rng = np.random.default_rng(seed)
    selected_blocks: list[int] = []
    for block_index in range(candidate_blocks):
        if len(selected_blocks) >= required_blocks:
            break

        remaining_candidates = candidate_blocks - block_index
        remaining_needed = required_blocks - len(selected_blocks)
        # 남은 후보 == 남은 필요 → 나머지 전부 선택 (suffix guard)
        if remaining_candidates == remaining_needed:
            selected_blocks.extend(range(block_index, candidate_blocks))
            break

        take_prob = compute_take_probability(
            required_blocks,
            candidate_blocks,
            remaining_candidates,
        )
        if float(rng.random()) < take_prob:
            selected_blocks.append(block_index)

    if len(selected_blocks) != required_blocks:
        raise RuntimeError(
            f"sequential planner failed to select exact block count: {len(selected_blocks)} != {required_blocks}"
        )
    return np.asarray(selected_blocks, dtype=np.int64), base_take_prob


def plan_sample(catalog: dict[str, object], seed: int, max_samples: int | None, *, split: str = "") -> SamplePlan:
    """순차 가산 샘플링으로 블록을 선택. 결정적이며 정확히 required_blocks개를 보장."""
    split_seed = seed ^ (hash(split) & 0xFFFFFFFF) if split else seed
    total_rows = int(catalog["total_rows"])
    total_blocks = int(math.ceil(total_rows / BLOCK_SIZE)) if total_rows > 0 else 0
    requested_rows = total_rows if max_samples is None else min(max_samples, total_rows)
    if requested_rows <= 0 or total_blocks == 0:
        return SamplePlan(
            requested_rows=0,
            effective_rows=0,
            required_blocks=0,
            total_blocks=total_blocks,
            candidate_blocks=0,
            planner_mode=SELECTION_PLANNER_MODE,
            base_take_prob=0.0,
            selected_blocks=np.empty((0,), dtype=np.int64),
            selected_bitmap=np.zeros(total_blocks, dtype=np.bool_),
            execution_block_count=0,
        )

    required_blocks = min(total_blocks, int(math.ceil(requested_rows / BLOCK_SIZE)))
    effective_rows = required_blocks * BLOCK_SIZE
    candidate_blocks = min(total_blocks, CANDIDATE_WINDOW_FACTOR * required_blocks)
    selected_blocks, base_take_prob = _select_blocks_sequentially(
        required_blocks=required_blocks,
        candidate_blocks=candidate_blocks,
        seed=split_seed,
    )

    selected_bitmap = np.zeros(total_blocks, dtype=np.bool_)
    selected_bitmap[selected_blocks] = True
    return SamplePlan(
        requested_rows=requested_rows,
        effective_rows=effective_rows,
        required_blocks=required_blocks,
        total_blocks=total_blocks,
        candidate_blocks=candidate_blocks,
        planner_mode=SELECTION_PLANNER_MODE,
        base_take_prob=base_take_prob,
        selected_blocks=selected_blocks,
        selected_bitmap=selected_bitmap,
        execution_block_count=(int(selected_blocks[-1]) + 1) if selected_blocks.size else 0,
    )


def _compress_runs_from_kinds(kinds: list[str], *, start_block: int = 0) -> list[ExecutionRun]:
    if not kinds:
        return []

    runs: list[ExecutionRun] = []
    current_kind = kinds[0]
    current_start = start_block
    for offset, kind in enumerate(kinds[1:], start=1):
        block_index = start_block + offset
        if kind == current_kind:
            continue
        runs.append(
            ExecutionRun(
                kind=current_kind,
                start_block=current_start,
                stop_block=block_index,
                block_count=block_index - current_start,
                total_rows=(block_index - current_start) * BLOCK_SIZE,
            )
        )
        current_kind = kind
        current_start = block_index

    stop_block = start_block + len(kinds)
    runs.append(
        ExecutionRun(
            kind=current_kind,
            start_block=current_start,
            stop_block=stop_block,
            block_count=stop_block - current_start,
            total_rows=(stop_block - current_start) * BLOCK_SIZE,
        )
    )
    return runs


def compress_frontier_runs(
    missing_bitmap: np.ndarray,
    *,
    start_block: int,
    stop_block: int,
) -> list[ExecutionRun]:
    if stop_block <= start_block:
        return []
    kinds = [
        "materialize" if bool(missing_bitmap[block_index]) else "skip"
        for block_index in range(start_block, stop_block)
    ]
    return _compress_runs_from_kinds(kinds, start_block=start_block)


def compress_execution_runs(
    selected_bitmap: np.ndarray,
    cached_bitmap: np.ndarray,
    *,
    stop_block: int,
) -> list[ExecutionRun]:
    """인접한 동일 종류 블록을 연속 실행 구간으로 병합."""
    if stop_block <= 0:
        return []

    kinds: list[str] = []
    for block_index in range(stop_block):
        if not bool(selected_bitmap[block_index]):
            kinds.append("skip")
            continue
        kinds.append("take_cached" if bool(cached_bitmap[block_index]) else "take_remote")
    return _compress_runs_from_kinds(kinds)


def build_cache_plan(
    sample_plan: SamplePlan,
    cached_bitmap: np.ndarray,
    *,
    frontier_block: int,
) -> CachePlan:
    """sample plan과 캐시 상태로부터 warmup 실행 계획 생성. 캐시 히트 시 frontier_runs 비어 있음."""
    selected_bitmap = sample_plan.selected_bitmap
    selected_blocks = sample_plan.selected_blocks
    hit_bitmap = np.logical_and(selected_bitmap, cached_bitmap)
    missing_bitmap = np.logical_and(selected_bitmap, ~cached_bitmap)
    cached_blocks = int(np.count_nonzero(hit_bitmap))
    missing_blocks = int(np.count_nonzero(missing_bitmap))
    cache_only = missing_blocks == 0
    execution_runs = compress_execution_runs(
        selected_bitmap,
        cached_bitmap,
        stop_block=sample_plan.execution_block_count,
    )
    if missing_blocks == 0:
        return CachePlan(
            sample_plan=sample_plan,
            hit_bitmap=hit_bitmap,
            missing_bitmap=missing_bitmap,
            cached_blocks=cached_blocks,
            missing_blocks=0,
            cache_only=True,
            execution_runs=execution_runs,
            frontier_runs=[],
            stream_start_block=frontier_block,
        )

    missing_indices = np.nonzero(missing_bitmap[selected_blocks])[0]
    highest_missing_block = int(selected_blocks[int(missing_indices[-1])])
    lowest_missing_block = int(selected_blocks[int(missing_indices[0])])
    stream_start_block = 0 if lowest_missing_block < frontier_block else frontier_block
    frontier_runs = compress_frontier_runs(
        missing_bitmap,
        start_block=stream_start_block,
        stop_block=highest_missing_block + 1,
    )
    return CachePlan(
        sample_plan=sample_plan,
        hit_bitmap=hit_bitmap,
        missing_bitmap=missing_bitmap,
        cached_blocks=cached_blocks,
        missing_blocks=missing_blocks,
        cache_only=False,
        execution_runs=execution_runs,
        frontier_runs=frontier_runs,
        stream_start_block=stream_start_block,
    )
