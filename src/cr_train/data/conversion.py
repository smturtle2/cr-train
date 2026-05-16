from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CACHE_BLOCK_SIZE, CACHE_LAYOUT_VERSION, LEGACY_CACHE_LAYOUT_VERSION
from .source import _build_block_descriptor, resolve_catalog_path, resolve_source_metadata_path
from .store import (
    BlockCachePaths,
    MappedBlockPayload,
    block_data_path,
    build_mapped_block_payload,
    read_json,
    resolve_block_cache_paths,
    write_json_atomic,
)
from .v15 import v15_block_is_cached, write_v15_block_arrays


@dataclass(frozen=True, slots=True)
class V14ToV15ConversionResult:
    source_root: Path
    destination_root: Path
    splits: tuple[str, ...]
    samples: int
    blocks: int
    raw_bytes: int
    compressed_bytes: int


def _source_signature(*, dataset_name: str, revision: str | None, split_sizes: dict[str, int]) -> str:
    payload = {
        "cache_layout_version": CACHE_LAYOUT_VERSION,
        "dataset_name": dataset_name,
        "revision": revision,
        "split_sizes": split_sizes,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def _find_v14_source_root(
    *,
    cache_root: Path,
    dataset_name: str,
    revision: str | None,
) -> tuple[Path, dict[str, Any]]:
    layout_root = cache_root / f"layout-v{LEGACY_CACHE_LAYOUT_VERSION}"
    if not layout_root.is_dir():
        raise FileNotFoundError(f"layout-v{LEGACY_CACHE_LAYOUT_VERSION} was not found: {layout_root}")

    matches: list[tuple[Path, dict[str, Any]]] = []
    for source_root in sorted(path for path in layout_root.iterdir() if path.is_dir()):
        metadata_path = resolve_source_metadata_path(source_root)
        if not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        if metadata.get("dataset_name") == dataset_name and metadata.get("revision") == revision:
            matches.append((source_root, metadata))

    if not matches:
        raise FileNotFoundError(
            f"no layout-v{LEGACY_CACHE_LAYOUT_VERSION} source matched "
            f"dataset_name={dataset_name!r}, revision={revision!r}"
        )
    if len(matches) > 1:
        roots = ", ".join(os.fspath(root) for root, _metadata in matches)
        raise RuntimeError(f"multiple layout-v{LEGACY_CACHE_LAYOUT_VERSION} sources matched: {roots}")
    return matches[0]


def _resolve_splits(descriptor: dict[str, Any], splits: Iterable[str] | None) -> tuple[str, ...]:
    split_sizes = descriptor.get("split_sizes", {})
    if splits is None:
        return tuple(str(split) for split in split_sizes)
    resolved = tuple(str(split) for split in splits)
    missing = [split for split in resolved if split not in split_sizes]
    if missing:
        raise KeyError(f"split(s) do not exist in v14 source descriptor: {', '.join(missing)}")
    return resolved


def _slice_payload_metadata(payload: Any, *, row_start: int, row_count: int) -> dict[str, Any]:
    row_end = row_start + row_count
    return {
        "row_count": row_count,
        "season": list(payload.season[row_start:row_end]),
        "scene": list(payload.scene[row_start:row_end]),
        "patch": list(payload.patch[row_start:row_end]),
        "sar_shape": [list(shape) for shape in payload.sar_shape[row_start:row_end]],
        "opt_shape": [list(shape) for shape in payload.opt_shape[row_start:row_end]],
    }


def _block_metadata(*, block: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "cache_key": str(block["cache_key"]),
        "split": split,
        "block_index": int(block["index"]),
        "shard_index": int(block["shard_index"]),
        "source_file": str(block["source_file"]),
        "row_groups": list(block["row_groups"]),
        "row_start": int(block["row_start"]),
        "row_count": int(block["row_count"]),
    }


def _load_v14_block_without_mutation(paths: BlockCachePaths, cache_key: str) -> MappedBlockPayload:
    block_path = block_data_path(paths, cache_key)
    try:
        payload_metadata = read_json(block_path / "payload.json")
        sar = np.load(block_path / "sar.npy", mmap_mode="r")
        cloudy = np.load(block_path / "cloudy.npy", mmap_mode="r")
        target = np.load(block_path / "target.npy", mmap_mode="r")
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"layout-v14 cached block payload is unreadable for {cache_key}; source cache was left untouched"
        ) from exc

    return build_mapped_block_payload(
        payload_metadata=payload_metadata,
        sar=sar,
        cloudy=cloudy,
        target=target,
    )


def convert_v14_cache_to_v15(
    *,
    source_cache_root: str | os.PathLike[str],
    destination_cache_root: str | os.PathLike[str] | None = None,
    dataset_name: str,
    revision: str | None,
    splits: Iterable[str] | None = None,
    overwrite: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> V14ToV15ConversionResult:
    """Convert a local layout-v14 cache into a local layout-v15 cache.

    This function only reads local v14 cache files and writes local v15 files. It does
    not download from or upload to B2.
    """
    source_cache_root_path = Path(source_cache_root)
    destination_cache_root_path = (
        source_cache_root_path
        if destination_cache_root is None
        else Path(destination_cache_root)
    )
    v14_source_root, v14_descriptor = _find_v14_source_root(
        cache_root=source_cache_root_path,
        dataset_name=dataset_name,
        revision=revision,
    )
    split_sizes = {str(split): int(size) for split, size in v14_descriptor["split_sizes"].items()}
    v15_signature = _source_signature(
        dataset_name=dataset_name,
        revision=revision,
        split_sizes=split_sizes,
    )
    v15_source_root = destination_cache_root_path / f"layout-v{CACHE_LAYOUT_VERSION}" / v15_signature
    v15_source_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        resolve_source_metadata_path(v15_source_root),
        {
            "cache_layout_version": CACHE_LAYOUT_VERSION,
            "dataset_name": dataset_name,
            "revision": revision,
            "source_signature": v15_signature,
            "split_sizes": split_sizes,
        },
    )

    resolved_splits = _resolve_splits(v14_descriptor, splits)
    total_samples = 0
    total_blocks = 0
    total_raw_bytes = 0
    total_compressed_bytes = 0

    for split in resolved_splits:
        v14_catalog = read_json(resolve_catalog_path(v14_source_root, split))
        v14_paths = resolve_block_cache_paths(v14_source_root, split)
        v15_blocks: list[dict[str, Any]] = []
        v15_block_row_counts: list[int] = []
        split_samples = 0

        for v14_block in v14_catalog.get("blocks", []):
            v14_payload = _load_v14_block_without_mutation(v14_paths, str(v14_block["cache_key"]))
            v14_row_count = len(v14_payload)
            for row_start in range(0, v14_row_count, CACHE_BLOCK_SIZE):
                row_count = min(CACHE_BLOCK_SIZE, v14_row_count - row_start)
                block = _build_block_descriptor(
                    dataset_name=dataset_name,
                    revision=revision,
                    split=split,
                    index=len(v15_blocks),
                    shard_index=int(v14_block["shard_index"]),
                    source_file=str(v14_block["source_file"]),
                    row_groups=tuple(int(value) for value in v14_block["row_groups"]),
                    row_start=row_start,
                    row_count=row_count,
                ).to_payload()
                cache_key = str(block["cache_key"])
                if overwrite or not v15_block_is_cached(v15_source_root, split, cache_key):
                    save_result = write_v15_block_arrays(
                        source_root=v15_source_root,
                        split=split,
                        cache_key=cache_key,
                        payload_metadata=_slice_payload_metadata(
                            v14_payload,
                            row_start=row_start,
                            row_count=row_count,
                        ),
                        metadata=_block_metadata(block=block, split=split),
                        sar=np.asarray(v14_payload.sar[row_start : row_start + row_count]),
                        cloudy=np.asarray(v14_payload.cloudy[row_start : row_start + row_count]),
                        target=np.asarray(v14_payload.target[row_start : row_start + row_count]),
                    )
                    total_compressed_bytes += save_result.payload_bytes
                total_raw_bytes += int(v14_payload.sar[row_start : row_start + row_count].nbytes)
                total_raw_bytes += int(v14_payload.cloudy[row_start : row_start + row_count].nbytes)
                total_raw_bytes += int(v14_payload.target[row_start : row_start + row_count].nbytes)
                v15_blocks.append(block)
                v15_block_row_counts.append(row_count)
                split_samples += row_count
                total_blocks += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "split": split,
                            "cache_key": cache_key,
                            "blocks": total_blocks,
                            "samples": total_samples + split_samples,
                        }
                    )

        write_json_atomic(
            resolve_catalog_path(v15_source_root, split),
            {
                "cache_layout_version": CACHE_LAYOUT_VERSION,
                "cache_block_size": CACHE_BLOCK_SIZE,
                "split": split,
                "total_rows": split_samples,
                "total_blocks": len(v15_blocks),
                "block_row_counts": v15_block_row_counts,
                "blocks": v15_blocks,
            },
        )
        total_samples += split_samples

    return V14ToV15ConversionResult(
        source_root=v14_source_root,
        destination_root=v15_source_root,
        splits=resolved_splits,
        samples=total_samples,
        blocks=total_blocks,
        raw_bytes=total_raw_bytes,
        compressed_bytes=total_compressed_bytes,
    )


__all__ = [
    "V14ToV15ConversionResult",
    "convert_v14_cache_to_v15",
]
