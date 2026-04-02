"""Block-selection bitmask algorithm visualization.

Demonstrates how the current stop-biased exact-k planner selects blocks
from the full logical block order by first sampling a stop block and then
drawing the remaining blocks from its prefix.

This is a synthetic planner demo. It does not load a real Parquet catalog,
so it uses the library's fixed block-size accounting constant.

Usage:
    uv run python examples/bitmask_sampling_demo.py
    uv run python examples/bitmask_sampling_demo.py --total-rows 107072 --requested-rows 2048 --seed 9

Output:
    - Configuration summary (fixed block size, total/required blocks)
    - Stop-block probability table and sampled stop
    - Prefix draw summary from the sampled stop block
    - Final selection bitmap (selected vs. skipped)
    - Selection efficiency statistics
"""

from __future__ import annotations

import argparse

import numpy as np

from cr_train.data import BLOCK_SIZE, trace_plan_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show how the deterministic block-selection bitmask is built.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--total-rows", type=int, default=107072, help="Total rows in the split.")
    parser.add_argument("--requested-rows", type=int, default=2048, help="Requested sample rows.")
    parser.add_argument("--seed", type=int, default=9, help="Sample-selection seed.")
    return parser.parse_args()


def _render_bitmask(bitmask: np.ndarray, *, width: int) -> str:
    return "".join("■" if bool(value) else "□" for value in bitmask[:width])


def _compact_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    head = max(1, (max_chars - 1) // 2)
    tail = max(1, max_chars - head - 1)
    return f"{text[:head]}…{text[-tail:]}"


def _format_index_ranges(values: list[int], *, max_ranges: int = 12) -> str:
    if not values:
        return "[]"
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    if len(ranges) <= max_ranges:
        return "[" + ", ".join(ranges) + "]"
    head = max(1, max_ranges // 2)
    tail = max(1, max_ranges - head)
    compact = ranges[:head] + ["..."] + ranges[-tail:]
    return "[" + ", ".join(compact) + "]"


def _build_stop_trace_lines(
    stop_candidates: list[int],
    stop_probabilities: list[float],
    *,
    sampled_stop_block: int | None,
    head: int = 12,
    tail: int = 4,
) -> list[str]:
    total = len(stop_candidates)
    if total <= head + tail + 1:
        display_indices = list(range(total))
    else:
        display_indices = list(range(head)) + list(range(total - tail, total))
        if sampled_stop_block is not None:
            sampled_index = stop_candidates.index(sampled_stop_block)
            display_indices.append(sampled_index)
        display_indices = sorted(set(display_indices))

    lines: list[str] = []
    previous_index: int | None = None
    for index in display_indices:
        if previous_index is not None and index != previous_index + 1:
            lines.append("  ...")
        stop_block = stop_candidates[index]
        probability = stop_probabilities[index]
        suffix = "  <<< sampled stop" if sampled_stop_block == stop_block else ""
        lines.append(f"  stop {stop_block:>4d}  p={probability:.4f}{suffix}")
        previous_index = index
    return lines


def build_selection_trace(
    *,
    total_rows: int,
    requested_rows: int,
    seed: int,
) -> dict[str, object]:
    trace = trace_plan_sample({"total_rows": total_rows}, seed=seed, max_samples=requested_rows)
    stop_trace_lines = _build_stop_trace_lines(
        trace.stop_candidates.tolist(),
        trace.stop_probabilities.tolist(),
        sampled_stop_block=trace.sampled_stop_block,
    )
    return {
        "total_blocks": trace.total_blocks,
        "required_blocks": trace.required_blocks,
        "planner_mode": trace.planner_mode,
        "stop_bias_alpha": trace.stop_bias_alpha,
        "sampled_stop_block": trace.sampled_stop_block,
        "prefix_draws": trace.prefix_draws.tolist(),
        "prefix_draw_summary": _format_index_ranges(trace.prefix_draws.tolist()),
        "selected_blocks": trace.selected_blocks.tolist(),
        "selected_bitmap": trace.selected_bitmap,
        "trace_lines": stop_trace_lines,
    }


def main() -> None:
    args = parse_args()
    result = build_selection_trace(
        total_rows=args.total_rows,
        requested_rows=args.requested_rows,
        seed=args.seed,
    )

    # --- Configuration ---
    print("=" * 60)
    print("  Block Selection Bitmask Demo")
    print("=" * 60)
    print()
    print(f"  block_rows          = {BLOCK_SIZE}")
    print(f"  total_rows          = {args.total_rows}")
    print(f"  requested_rows      = {args.requested_rows}")
    print(f"  seed                = {args.seed}")
    print()
    print(f"  total_blocks        = {result['total_blocks']}")
    print(f"  required_blocks     = {result['required_blocks']}")
    print(f"  planner_mode        = {result['planner_mode']}")
    print(f"  stop_bias_alpha     = {result['stop_bias_alpha']:.4f}")
    print(f"  sampled_stop_block  = {result['sampled_stop_block']}")
    print()

    # --- Stop distribution trace ---
    print("-" * 60)
    print("  Stop-block probability trace")
    print("-" * 60)
    print()
    for line in result["trace_lines"]:
        print(line)
    print()
    print(f"  prefix_draws        = {result['prefix_draw_summary']}")
    print()

    # --- Selection bitmap ---
    print("-" * 60)
    print("  Selection bitmap (first N blocks)")
    print("  ■ = selected    □ = skipped")
    print("-" * 60)
    print()
    bitmap_width = min(int(result["total_blocks"]), 80)
    bitmap = _render_bitmask(result["selected_bitmap"], width=int(result["total_blocks"]))
    print(f"  {_compact_text(bitmap, max_chars=bitmap_width)}")
    print()

    # --- Summary statistics ---
    selected = list(result["selected_blocks"])
    required = int(result["required_blocks"])
    total = int(result["total_blocks"])
    span = (max(selected) - min(selected) + 1) if selected else 0

    print("-" * 60)
    print("  Summary")
    print("-" * 60)
    print()
    print(f"  selected_blocks     = {len(selected)} / {required} required")
    print(f"  total_blocks        = {total}")
    print(f"  selection_span      = {span} blocks (contiguous range covering all selected)")
    print(f"  selection_density   = {len(selected) / max(span, 1):.2%} within span")
    print(f"  effective_rows_est  = {len(selected) * BLOCK_SIZE}")
    print()


if __name__ == "__main__":
    main()
