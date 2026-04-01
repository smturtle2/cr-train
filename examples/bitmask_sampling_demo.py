"""Block-selection bitmask algorithm visualization.

Demonstrates how the sequential additive exact-k planner selects blocks
from a candidate window. The take probability increases as the remaining
window shrinks, guaranteeing the exact required block count.

Usage:
    uv run python examples/bitmask_sampling_demo.py
    uv run python examples/bitmask_sampling_demo.py --total-rows 107072 --requested-rows 2048 --seed 9

Output:
    - Configuration summary (block size, total/required/candidate blocks)
    - Per-block trace: remaining candidates, take probability, random draw, decision
    - Final selection bitmap (selected vs. skipped)
    - Selection efficiency statistics
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from cr_train.data import BLOCK_SIZE, compute_base_take_probability, compute_take_probability


CANDIDATE_WINDOW_FACTOR = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show how the canonical block-selection bitmask is built.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--total-rows", type=int, default=107072, help="Total rows in the split.")
    parser.add_argument("--requested-rows", type=int, default=2048, help="Requested sample rows.")
    parser.add_argument("--seed", type=int, default=9, help="Sample-selection seed.")
    return parser.parse_args()


def _render_bitmask(bitmask: np.ndarray, *, width: int) -> str:
    return "".join("■" if bool(value) else "□" for value in bitmask[:width])


def build_selection_trace(
    *,
    total_rows: int,
    requested_rows: int,
    seed: int,
) -> dict[str, object]:
    total_blocks = int(math.ceil(total_rows / BLOCK_SIZE)) if total_rows > 0 else 0
    required_blocks = min(total_blocks, int(math.ceil(requested_rows / BLOCK_SIZE))) if total_blocks > 0 else 0
    candidate_blocks = min(total_blocks, CANDIDATE_WINDOW_FACTOR * required_blocks)
    base_take_prob = compute_base_take_probability(required_blocks, candidate_blocks)

    selected_blocks: list[int] = []
    trace_lines: list[str] = []
    rng = np.random.default_rng(seed)

    for block_index in range(candidate_blocks):
        if len(selected_blocks) >= required_blocks:
            break

        remaining_candidates = candidate_blocks - block_index
        remaining_needed = required_blocks - len(selected_blocks)
        if remaining_candidates == remaining_needed:
            suffix = list(range(block_index, candidate_blocks))
            selected_blocks.extend(suffix)
            trace_lines.append(
                f"  block {block_index:>4d}  remaining={remaining_candidates:>4d}  "
                f"** guard take ** suffix={suffix}"
            )
            break

        take_prob = compute_take_probability(
            required_blocks,
            candidate_blocks,
            remaining_candidates,
        )
        sampled = float(rng.random())
        take = sampled < take_prob
        if take:
            selected_blocks.append(block_index)

        marker = ">>>" if take else "   "
        trace_lines.append(
            f"  block {block_index:>4d}  remaining={remaining_candidates:>4d}  "
            f"p={take_prob:.4f}  u={sampled:.4f}  {marker} {'TAKE' if take else 'skip'}"
        )

    selected_bitmap = np.zeros(total_blocks, dtype=np.bool_)
    if selected_blocks:
        selected_bitmap[np.asarray(selected_blocks, dtype=np.int64)] = True

    return {
        "total_blocks": total_blocks,
        "required_blocks": required_blocks,
        "candidate_blocks": candidate_blocks,
        "base_take_prob": base_take_prob,
        "selected_blocks": selected_blocks,
        "selected_bitmap": selected_bitmap,
        "trace_lines": trace_lines,
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
    print(f"  block_size          = {BLOCK_SIZE}")
    print(f"  total_rows          = {args.total_rows}")
    print(f"  requested_rows      = {args.requested_rows}")
    print(f"  seed                = {args.seed}")
    print()
    print(f"  total_blocks        = {result['total_blocks']}")
    print(f"  required_blocks     = {result['required_blocks']}")
    print(f"  candidate_blocks    = {result['candidate_blocks']}")
    print(f"  base_take_prob      = {result['base_take_prob']:.4f}")
    print()

    # --- Per-block trace ---
    print("-" * 60)
    print("  Per-block sampling trace")
    print("-" * 60)
    print()
    for line in result["trace_lines"]:
        print(line)
    print()

    # --- Selection bitmap ---
    print("-" * 60)
    print("  Selection bitmap (first N blocks)")
    print("  ■ = selected    □ = skipped")
    print("-" * 60)
    print()
    bitmap_width = min(int(result["candidate_blocks"]), 80)
    print(f"  {_render_bitmask(result['selected_bitmap'], width=bitmap_width)}")
    print()

    # --- Summary statistics ---
    selected = result["selected_blocks"]
    required = int(result["required_blocks"])
    candidate = int(result["candidate_blocks"])
    total = int(result["total_blocks"])
    span = (max(selected) - min(selected) + 1) if selected else 0

    print("-" * 60)
    print("  Summary")
    print("-" * 60)
    print()
    print(f"  selected_blocks     = {len(selected)} / {required} required")
    print(f"  candidate_window    = {candidate} / {total} total blocks")
    print(f"  selection_span      = {span} blocks (contiguous range covering all selected)")
    print(f"  selection_density   = {len(selected) / max(span, 1):.2%} within span")
    print(f"  effective_rows      = {len(selected) * BLOCK_SIZE}")
    print()


if __name__ == "__main__":
    main()
