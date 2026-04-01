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
                f"block={block_index:02d} guard take suffix={suffix}"
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
        trace_lines.append(
            f"block={block_index:02d} remaining={remaining_candidates:02d} "
            f"p={take_prob:.4f} u={sampled:.4f} take={str(take).lower()}"
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

    print(f"block_size={BLOCK_SIZE}")
    print(f"total_blocks={result['total_blocks']}")
    print(f"required_blocks={result['required_blocks']}")
    print(f"candidate_blocks={result['candidate_blocks']}")
    print(f"base_take_prob={result['base_take_prob']:.4f}")
    print()
    for line in result["trace_lines"]:
        print(line)
    print()
    print(f"selected_blocks={result['selected_blocks']}")
    print(
        "selected_bitmap="
        + _render_bitmask(
            result["selected_bitmap"],
            width=int(result["total_blocks"]),
        )
    )


if __name__ == "__main__":
    main()
