"""Measure SEN12MS-CR sample throughput without training.

Usage:
    uv run python examples/measure_sample_throughput.py \
      --streaming \
      --split train \
      --max-samples 1024 \
      --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cr_train.data import DATASET_ID, build_dataloader, resolve_num_workers
from cr_train.data.hf_v2 import resolve_dataset_root
from cr_train.data.dataset import prepare_split


def parse_max_samples(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"none", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("sample counts must be positive or 'none'")
    return parsed


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_num_workers(value: str) -> int | str:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("num_workers must be zero or greater, or 'auto'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure sample throughput without model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream HF v2 crpack blocks instead of using a persistent local dataset.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Persistent local HF v2 dataset directory used when --no-streaming is set.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
        help="Dataset split to iterate.",
    )
    parser.add_argument(
        "--max-samples",
        type=parse_max_samples,
        default=64,
        help="Requested rows; converted to HF v2 block count, or 'none'/'full'.",
    )
    parser.add_argument("--batch-size", type=parse_positive_int, default=8)
    parser.add_argument(
        "--num-workers",
        type=parse_num_workers,
        default="auto",
        help="PyTorch DataLoader worker process count.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch index used for deterministic block order.",
    )
    return parser.parse_args()


def batch_size_of(batch: dict[str, Any]) -> int:
    return int(batch["sar"].shape[0])


def main() -> None:
    args = parse_args()
    dataloader_workers = resolve_num_workers(args.num_workers)
    dataset_root = resolve_dataset_root(args.dataset_dir)

    pipeline_started_at = time.perf_counter()
    prepare_started_at = time.perf_counter()
    prepared = prepare_split(
        split=args.split,
        dataset_name=DATASET_ID,
        revision=None,
        max_samples=args.max_samples,
        seed=args.seed,
        epoch=args.epoch,
        training=False,
        dataset_root=None if args.streaming else dataset_root,
        streaming=args.streaming,
    )
    prepare_split_sec = time.perf_counter() - prepare_started_at

    dataloader_started_at = time.perf_counter()
    loader = build_dataloader(
        prepared,
        batch_size=args.batch_size,
        num_workers=dataloader_workers,
        training=False,
        seed=args.seed,
        epoch=args.epoch,
        include_metadata=False,
        pin_memory=False,
    )
    dataloader_build_sec = time.perf_counter() - dataloader_started_at

    iterator_started_at = time.perf_counter()
    iterator = iter(loader)
    iterator_create_sec = time.perf_counter() - iterator_started_at
    total_started_at = time.perf_counter()
    try:
        first_batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("no samples were produced") from exc
    first_batch_wait_sec = time.perf_counter() - total_started_at
    first_batch_samples = batch_size_of(first_batch)

    num_batches = 0
    num_samples = 0
    measured_started_at = time.perf_counter()
    for batch in iterator:
        num_batches += 1
        num_samples += batch_size_of(batch)
    elapsed_sec = time.perf_counter() - measured_started_at
    total_elapsed_sec = time.perf_counter() - total_started_at
    pipeline_elapsed_sec = time.perf_counter() - pipeline_started_at

    samples_per_sec = num_samples / elapsed_sec if elapsed_sec > 0 else 0.0
    batches_per_sec = num_batches / elapsed_sec if elapsed_sec > 0 else 0.0
    pipeline_samples = num_samples + first_batch_samples
    pipeline_samples_per_sec = (
        pipeline_samples / pipeline_elapsed_sec if pipeline_elapsed_sec > 0 else 0.0
    )
    result = {
        "kind": "sample_throughput",
        "streaming": args.streaming,
        "dataset_dir": None if args.streaming else str(dataset_root),
        "split": args.split,
        "requested_rows": args.max_samples,
        "prepared_rows": prepared.num_examples,
        "batch_size": args.batch_size,
        "dataloader_workers": dataloader_workers,
        "warmup_excluded": True,
        "first_batch_wait_sec": first_batch_wait_sec,
        "first_batch_samples": first_batch_samples,
        "prepare_split_sec": prepare_split_sec,
        "dataloader_build_sec": dataloader_build_sec,
        "iterator_create_sec": iterator_create_sec,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "elapsed_sec": elapsed_sec,
        "samples_per_sec": samples_per_sec,
        "batches_per_sec": batches_per_sec,
        "total_batches": num_batches + 1,
        "total_samples": num_samples + first_batch_samples,
        "total_elapsed_sec": total_elapsed_sec,
        "pipeline_elapsed_sec": pipeline_elapsed_sec,
        "pipeline_samples_per_sec": pipeline_samples_per_sec,
    }
    source = "streaming" if args.streaming else "local"
    print(
        f"{source} {args.split}: {pipeline_samples_per_sec:.2f} samples/sec pipeline "
        f"({pipeline_samples} samples, {pipeline_elapsed_sec:.2f}s from prepare_split to drained); "
        f"{samples_per_sec:.2f} samples/sec after first batch "
        f"({num_samples} measured samples, {num_batches} batches, {elapsed_sec:.2f}s; "
        f"DataLoader workers {dataloader_workers}; "
        f"first batch wait {first_batch_wait_sec:.2f}s)"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
