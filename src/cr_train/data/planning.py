from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .constants import BLOCK_SIZE, CANDIDATE_WINDOW_FACTOR, STOP_BIAS_ALPHA, STOP_BIAS_MIX


SELECTION_PLANNER_MODE = "stop_biased_exact_k"


@dataclass(slots=True)
class SamplePlan:
    """블록 단위 샘플 선택 결과. selected_bitmap이 어떤 블록을 사용할지 결정."""

    requested_rows: int
    effective_rows: int
    required_blocks: int
    total_blocks: int
    candidate_blocks: int
    planner_mode: str
    stop_bias_alpha: float
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
    cached_selected_blocks: int
    selected_missing_blocks: int
    extension_blocks: int
    cache_only: bool
    frontier_before: int
    frontier_after: int


@dataclass(slots=True)
class SelectionTrace:
    total_blocks: int
    requested_rows: int
    required_blocks: int
    candidate_blocks: int
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
    candidate_blocks: int,
    stop_block: int,
    *,
    stop_bias_alpha: float = STOP_BIAS_ALPHA,
) -> float:
    """candidate window 내부 stop block 확률 질량 함수."""
    if required_blocks <= 0 or candidate_blocks <= 0:
        return 0.0
    min_stop = required_blocks - 1
    max_stop = candidate_blocks - 1
    if stop_block < min_stop or stop_block > max_stop:
        return 0.0
    positions = np.arange(min_stop, max_stop + 1, dtype=np.float64)
    denom = max(candidate_blocks - required_blocks, 1)
    weights = np.exp(-stop_bias_alpha * ((positions - min_stop) / denom))
    weights /= float(np.sum(weights))
    uniform = np.full_like(weights, 1.0 / len(weights))
    mixed_weights = (STOP_BIAS_MIX * weights) + ((1.0 - STOP_BIAS_MIX) * uniform)
    weights_sum = float(np.sum(mixed_weights))
    if weights_sum <= 0.0:
        return 0.0
    return float(mixed_weights[int(stop_block - min_stop)] / weights_sum)


def compute_base_take_probability(required_blocks: int, candidate_blocks: int) -> float:
    """호환용 legacy helper. 새 planner는 사용하지 않는다."""
    if required_blocks <= 0 or candidate_blocks <= 0:
        return 0.0
    return min(1.0, required_blocks / candidate_blocks)


def compute_take_probability(
    required_blocks: int,
    candidate_blocks: int,
    remaining_candidates: int,
) -> float:
    """호환용 legacy helper. 새 planner는 사용하지 않는다."""
    if required_blocks <= 0 or candidate_blocks <= 0 or remaining_candidates <= 0:
        return 0.0
    base_take_prob = compute_base_take_probability(required_blocks, candidate_blocks)
    dynamic_bonus = max(0.0, (required_blocks / remaining_candidates) - base_take_prob)
    return min(1.0, base_take_prob + dynamic_bonus)


def _derive_split_seed(seed: int, split: str) -> int:
    if not split:
        return int(seed)
    digest = hashlib.sha256(split.encode("utf-8")).digest()
    split_bits = int.from_bytes(digest[:8], "big")
    return int(seed) ^ split_bits


def _sample_stop_block(
    *,
    required_blocks: int,
    candidate_blocks: int,
    rng: np.random.Generator,
    stop_bias_alpha: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    min_stop = required_blocks - 1
    max_stop = candidate_blocks - 1
    if min_stop >= max_stop:
        stop_candidates = np.asarray([min_stop], dtype=np.int64)
        probabilities = np.asarray([1.0], dtype=np.float64)
        return min_stop, stop_candidates, probabilities
    stop_candidates = np.arange(min_stop, max_stop + 1, dtype=np.int64)
    probabilities = np.asarray(
        [
            compute_stop_probability(
                required_blocks,
                candidate_blocks,
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
    candidate_blocks: int,
    seed: int,
    stop_bias_alpha: float,
) -> tuple[np.ndarray, float]:
    if required_blocks <= 0 or candidate_blocks <= 0:
        return np.empty((0,), dtype=np.int64), stop_bias_alpha
    if required_blocks >= candidate_blocks:
        return np.arange(candidate_blocks, dtype=np.int64), stop_bias_alpha

    rng = np.random.default_rng(seed)
    stop_block, _, _ = _sample_stop_block(
        required_blocks=required_blocks,
        candidate_blocks=candidate_blocks,
        rng=rng,
        stop_bias_alpha=stop_bias_alpha,
    )
    if required_blocks == 1:
        return np.asarray([stop_block], dtype=np.int64), stop_bias_alpha
    prefix_blocks = rng.choice(stop_block, size=required_blocks - 1, replace=False).astype(np.int64)
    selected_blocks = np.sort(np.concatenate((prefix_blocks, np.asarray([stop_block], dtype=np.int64))))
    return selected_blocks, stop_bias_alpha


def trace_plan_sample(
    catalog: dict[str, object],
    seed: int,
    max_samples: int | None,
    *,
    split: str = "",
) -> SelectionTrace:
    """예제/디버깅용 planner trace. 실제 sampling 로직을 그대로 재사용한다."""
    split_seed = _derive_split_seed(seed, split)
    total_rows = int(catalog["total_rows"])
    total_blocks = int(math.ceil(total_rows / BLOCK_SIZE)) if total_rows > 0 else 0
    requested_rows = total_rows if max_samples is None else min(max_samples, total_rows)
    if requested_rows <= 0 or total_blocks == 0:
        return SelectionTrace(
            total_blocks=total_blocks,
            requested_rows=0,
            required_blocks=0,
            candidate_blocks=0,
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
    candidate_blocks = min(total_blocks, CANDIDATE_WINDOW_FACTOR * required_blocks)
    if required_blocks >= candidate_blocks:
        selected_blocks = np.arange(candidate_blocks, dtype=np.int64)
        selected_bitmap = np.zeros(total_blocks, dtype=np.bool_)
        selected_bitmap[selected_blocks] = True
        return SelectionTrace(
            total_blocks=total_blocks,
            requested_rows=requested_rows,
            required_blocks=required_blocks,
            candidate_blocks=candidate_blocks,
            planner_mode=SELECTION_PLANNER_MODE,
            stop_bias_alpha=STOP_BIAS_ALPHA,
            stop_candidates=np.asarray([candidate_blocks - 1], dtype=np.int64),
            stop_probabilities=np.asarray([1.0], dtype=np.float64),
            sampled_stop_block=(candidate_blocks - 1) if candidate_blocks > 0 else None,
            prefix_draws=selected_blocks[:-1],
            selected_blocks=selected_blocks,
            selected_bitmap=selected_bitmap,
            execution_block_count=(int(selected_blocks[-1]) + 1) if selected_blocks.size else 0,
        )

    rng = np.random.default_rng(split_seed)
    sampled_stop_block, stop_candidates, stop_probabilities = _sample_stop_block(
        required_blocks=required_blocks,
        candidate_blocks=candidate_blocks,
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
        candidate_blocks=candidate_blocks,
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


def plan_sample(catalog: dict[str, object], seed: int, max_samples: int | None, *, split: str = "") -> SamplePlan:
    """stop-biased exact-k 샘플링으로 블록을 선택. 결정적이며 정확히 required_blocks개를 보장."""
    split_seed = _derive_split_seed(seed, split)
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
            stop_bias_alpha=STOP_BIAS_ALPHA,
            selected_blocks=np.empty((0,), dtype=np.int64),
            selected_bitmap=np.zeros(total_blocks, dtype=np.bool_),
            execution_block_count=0,
        )

    required_blocks = min(total_blocks, int(math.ceil(requested_rows / BLOCK_SIZE)))
    effective_rows = required_blocks * BLOCK_SIZE
    candidate_blocks = min(total_blocks, CANDIDATE_WINDOW_FACTOR * required_blocks)
    selected_blocks, stop_bias_alpha = _select_blocks_stop_biased(
        required_blocks=required_blocks,
        candidate_blocks=candidate_blocks,
        seed=split_seed,
        stop_bias_alpha=STOP_BIAS_ALPHA,
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
        stop_bias_alpha=stop_bias_alpha,
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
    """sample plan과 contiguous prefix 캐시 상태로부터 warmup 실행 계획 생성."""
    selected_bitmap = sample_plan.selected_bitmap
    cached_selected_blocks = int(np.count_nonzero(np.logical_and(selected_bitmap, cached_bitmap)))
    selected_missing_blocks = int(np.count_nonzero(np.logical_and(selected_bitmap, ~cached_bitmap)))
    frontier_before = max(0, int(frontier_block))
    frontier_after = max(frontier_before, sample_plan.execution_block_count)
    extension_blocks = max(0, frontier_after - frontier_before)
    cache_only = extension_blocks == 0
    if cache_only:
        return CachePlan(
            sample_plan=sample_plan,
            cached_selected_blocks=cached_selected_blocks,
            selected_missing_blocks=selected_missing_blocks,
            extension_blocks=0,
            cache_only=True,
            frontier_before=frontier_before,
            frontier_after=frontier_after,
        )

    return CachePlan(
        sample_plan=sample_plan,
        cached_selected_blocks=cached_selected_blocks,
        selected_missing_blocks=selected_missing_blocks,
        extension_blocks=extension_blocks,
        cache_only=False,
        frontier_before=frontier_before,
        frontier_after=frontier_after,
    )
