"""Block-selection bitmask algorithm visualization.

Demonstrates how the current uniform exact-k planner selects logical blocks
from the full logical block order.

This is a synthetic planner demo. It does not load a real Parquet catalog,
so it uses the library's fixed block-size accounting constant.

Usage:
    uv run python examples/bitmask_sampling_demo.py
    uv run python examples/bitmask_sampling_demo.py --total-rows 107072 --requested-rows 2048 --seed 9

Output:
    - Configuration summary (fixed block size, total/required blocks)
    - Raw uniform draw order
    - Final selected block indices
    - Final selection bitmap (selected vs. skipped)
    - Selection efficiency statistics
"""

from __future__ import annotations

import argparse

import numpy as np

from cr_train.data import BLOCK_SIZE, trace_plan_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show how the deterministic uniform exact-k bitmask is built.",
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


def build_selection_trace(
    *,
    total_rows: int,
    requested_rows: int,
    seed: int,
) -> dict[str, object]:
    trace = trace_plan_sample({"total_rows": total_rows}, seed=seed, max_samples=requested_rows)
    return {
        "total_blocks": trace.total_blocks,
        "required_blocks": trace.required_blocks,
        "planner_mode": trace.planner_mode,
        "draw_order": trace.draw_order.tolist(),
        "draw_order_summary": _format_index_ranges(trace.draw_order.tolist()),
        "selected_blocks": trace.selected_blocks.tolist(),
        "selected_blocks_summary": _format_index_ranges(trace.selected_blocks.tolist()),
        "selected_bitmap": trace.selected_bitmap,
    }


def main() -> None:
    args = parse_args()
    result = build_selection_trace(
        total_rows=args.total_rows,
        requested_rows=args.requested_rows,
        seed=args.seed,
    )

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

    print()
    print("--- Uniform exact-k draw ---")
    print()
    print(f"  draw_order          = {result['draw_order_summary']}")
    print(f"  selected_blocks     = {result['selected_blocks_summary']}")

    print()
    print("--- Selection bitmap  (■ selected, □ skipped) ---")
    print()
    bitmap_width = min(int(result["total_blocks"]), 80)
    bitmap = _render_bitmask(result["selected_bitmap"], width=int(result["total_blocks"]))
    print(f"  {_compact_text(bitmap, max_chars=bitmap_width)}")

    selected = list(result["selected_blocks"])
    required = int(result["required_blocks"])
    total = int(result["total_blocks"])
    span = (max(selected) - min(selected) + 1) if selected else 0

    print()
    print("--- Summary ---")
    print()
    print(f"  selected_blocks     = {len(selected)} / {required} required")
    print(f"  total_blocks        = {total}")
    print(f"  selection_span      = {span} blocks (contiguous range covering all selected)")
    print(f"  selection_density   = {len(selected) / max(span, 1):.2%} within span")
    print(f"  effective_rows_est  = {len(selected) * BLOCK_SIZE}")
    print()


if __name__ == "__main__":
    main()
