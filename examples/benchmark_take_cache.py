from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cr_train.take_cache import take_rows_official, take_rows_with_prefix_cache


def parse_csv_ints(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one integer")
    parsed = [int(part) for part in parts]
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all integers must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark official HF take against the local prefix-cache prototype.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-sizes", type=parse_csv_ints, default=[64, 256, 1024, 2048], help="Comma-separated list.")
    parser.add_argument("--buffer-sizes", type=parse_csv_ints, default=[256, 1024], help="Comma-separated list.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="Hermanni/sen12mscr")
    parser.add_argument("--cache-dir", default="runs/take-cache-bench")
    parser.add_argument("--clear-cache", action="store_true")
    return parser.parse_args()


def run_mode(
    *,
    mode: str,
    split: str,
    sample_size: int,
    buffer_size: int,
    seed: int,
    cache_dir: Path,
    dataset_name: str,
) -> dict[str, object]:
    if mode == "official":
        result = take_rows_official(
            split=split,
            seed=seed,
            sample_size=sample_size,
            buffer_size=buffer_size,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
        )
    else:
        result = take_rows_with_prefix_cache(
            split=split,
            seed=seed,
            sample_size=sample_size,
            buffer_size=buffer_size,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
        )

    record: dict[str, object] = {
        "mode": mode,
        "split": split,
        "sample_size": sample_size,
        "buffer_size": buffer_size,
        "seed": seed,
        "elapsed_sec": result.elapsed_sec,
        "digest": result.digest,
        "rows": len(result.rows),
    }
    if result.cache_stats is not None:
        record["cache_stats"] = {
            "required_prefix_rows": result.cache_stats.required_prefix_rows,
            "cached_prefix_rows_before": result.cache_stats.cached_prefix_rows_before,
            "cached_prefix_rows_after": result.cache_stats.cached_prefix_rows_after,
            "fetched_prefix_rows": result.cache_stats.fetched_prefix_rows,
            "fetched_shards": result.cache_stats.fetched_shards,
            "ensure_prefix_sec": result.cache_stats.elapsed_sec,
        }
    return record


def main() -> None:
    args = parse_args()
    base_cache_dir = Path(args.cache_dir)
    if args.clear_cache and base_cache_dir.exists():
        shutil.rmtree(base_cache_dir)

    scenarios = [
        ("cold", args.seed, None),
        ("warm_same_seed", args.seed, None),
        ("warm_other_seed", args.seed + 1, None),
    ]

    for buffer_size in args.buffer_sizes:
        for sample_size in args.sample_sizes:
            prototype_cache_dir = base_cache_dir / "prototype" / f"buffer-{buffer_size}"
            official_cache_dir = base_cache_dir / "official" / f"buffer-{buffer_size}"
            larger_sample_size = sample_size * 2
            scenario_specs = list(scenarios)
            scenario_specs.append(("warm_larger_n", args.seed, larger_sample_size))

            official_records: dict[str, dict[str, object]] = {}
            prototype_records: dict[str, dict[str, object]] = {}
            for scenario_name, seed, scenario_sample_size in scenario_specs:
                current_sample_size = scenario_sample_size or sample_size
                official_record = run_mode(
                    mode="official",
                    split=args.split,
                    sample_size=current_sample_size,
                    buffer_size=buffer_size,
                    seed=seed,
                    cache_dir=official_cache_dir,
                    dataset_name=args.dataset_name,
                )
                official_record["scenario"] = scenario_name
                print(json.dumps(official_record, sort_keys=True))
                official_records[scenario_name] = official_record

                prototype_record = run_mode(
                    mode="prototype",
                    split=args.split,
                    sample_size=current_sample_size,
                    buffer_size=buffer_size,
                    seed=seed,
                    cache_dir=prototype_cache_dir,
                    dataset_name=args.dataset_name,
                )
                prototype_record["scenario"] = scenario_name
                prototype_record["matches_official"] = prototype_record["digest"] == official_record["digest"]
                print(json.dumps(prototype_record, sort_keys=True))
                prototype_records[scenario_name] = prototype_record

            summary = {
                "buffer_size": buffer_size,
                "base_sample_size": sample_size,
                "official": {
                    name: record["elapsed_sec"]
                    for name, record in official_records.items()
                },
                "prototype": {
                    name: record["elapsed_sec"]
                    for name, record in prototype_records.items()
                },
                "all_match": all(
                    bool(prototype_records[name]["matches_official"])
                    for name in prototype_records
                ),
            }
            print(json.dumps({"summary": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
