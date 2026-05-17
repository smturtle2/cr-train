"""Measure SEN12MS-CR cache sample throughput without training.

Usage:
    uv run python examples/measure_sample_throughput.py \
      --cache-src B2 \
      --cache-dir cache \
      --split train \
      --max-samples 64 \
      --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cr_train.data import DATASET_ID, build_dataloader, resolve_num_workers
from cr_train.data.cache_backend import (
    CacheSource,
    b2_download_worker_count,
    normalize_cache_src,
)
from cr_train.data.dataset import prepare_split
from cr_train.data.store import remove_tree, resolve_cache_root


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
        description="Measure cache sample throughput without model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-src",
        choices=("local", "B2"),
        default="local",
        help="Cache source. local uses a filesystem cache; B2 uses an object-store prefix.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Local cache root for local mode, or B2 bucket prefix for B2 mode.",
    )
    parser.add_argument(
        "--b2-staging-dir",
        default=None,
        help="Local staging directory for B2 blocks. B2 mode only.",
    )
    parser.add_argument(
        "--b2-staging-max-blocks",
        type=parse_positive_int,
        default=80,
        help="Maximum number of B2 blocks staged locally ahead of consumption.",
    )
    parser.add_argument(
        "--b2-download-workers",
        type=parse_positive_int,
        default=b2_download_worker_count(),
        help="Number of internal B2 object download workers. This is separate from DataLoader workers.",
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
        help="Requested rows; converted to cache-block count, or 'none'/'full'.",
    )
    parser.add_argument("--batch-size", type=parse_positive_int, default=8)
    parser.add_argument(
        "--num-workers",
        type=parse_num_workers,
        default="auto",
        help="PyTorch DataLoader worker process count. This is separate from B2 download workers.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch index used for deterministic block order.",
    )
    return parser.parse_args()


def resolve_cache_location(cache_src: CacheSource, cache_dir: str | None) -> Path:
    if cache_src == "B2":
        return Path(cache_dir) if cache_dir is not None else Path("cache")
    return resolve_cache_root(cache_dir)


def batch_size_of(batch: dict[str, Any]) -> int:
    return int(batch["sar"].shape[0])


def main() -> None:
    args = parse_args()
    cache_src = normalize_cache_src(args.cache_src)
    cache_root = resolve_cache_location(cache_src, args.cache_dir)
    dataloader_workers = resolve_num_workers(args.num_workers)
    b2_staging_dir = (
        Path(args.b2_staging_dir)
        if args.b2_staging_dir is not None
        else Path.home() / ".cache" / "cr-train" / "b2-staging"
    )
    b2_staging_max_blocks = max(args.b2_staging_max_blocks, dataloader_workers + 1)
    if cache_src == "B2":
        remove_tree(b2_staging_dir)

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
        cache_root=cache_root,
        cache_src=cache_src,
        b2_staging_dir=b2_staging_dir if cache_src == "B2" else None,
        b2_staging_max_blocks=b2_staging_max_blocks,
        b2_download_workers=args.b2_download_workers if cache_src == "B2" else None,
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
        "cache_src": cache_src,
        "cache_dir": str(cache_root),
        "split": args.split,
        "requested_rows": args.max_samples,
        "prepared_rows": prepared.num_examples,
        "batch_size": args.batch_size,
        "dataloader_workers": dataloader_workers,
        "b2_download_workers": args.b2_download_workers if cache_src == "B2" else None,
        "b2_staging_dir": str(b2_staging_dir) if cache_src == "B2" else None,
        "b2_staging_max_blocks": b2_staging_max_blocks if cache_src == "B2" else None,
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
    print(
        f"{cache_src} {args.split}: {pipeline_samples_per_sec:.2f} samples/sec pipeline "
        f"({pipeline_samples} samples, {pipeline_elapsed_sec:.2f}s from prepare_split to drained); "
        f"{samples_per_sec:.2f} samples/sec after first batch "
        f"({num_samples} measured samples, {num_batches} batches, {elapsed_sec:.2f}s; "
        f"DataLoader workers {dataloader_workers}; "
        f"B2 download workers {args.b2_download_workers if cache_src == 'B2' else 'n/a'}; "
        f"B2 staging buffer {b2_staging_max_blocks if cache_src == 'B2' else 'n/a'} blocks; "
        f"first batch wait {first_batch_wait_sec:.2f}s)"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
