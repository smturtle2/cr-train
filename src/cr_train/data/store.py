from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc
from datasets import Dataset, disable_progress_bars, enable_progress_bars, is_progress_bar_enabled

from .constants import BLOCK_SIZE, DATA_COLUMNS, DEFAULT_DATASET_SEED, LOCK_POLL_INTERVAL_SECONDS, LOCK_TIMEOUT_SECONDS, OPTICAL_CHANNELS, SAR_CHANNELS


@dataclass(slots=True)
class SplitBlockCacheState:
    """블록 캐시의 영속 상태. canonical_frontier_block이 스트림 재개 위치를 추적."""

    dataset_seed: int
    shuffle_buffer_size: int
    block_size: int
    total_rows: int
    total_blocks: int
    canonical_frontier_block: int


@dataclass(slots=True)
class SplitBlockCache:
    """split 블록 캐시의 인메모리 표현. cached 비트맵으로 블록별 캐시 여부 관리."""

    state: SplitBlockCacheState
    cached: np.ndarray
    chunk_ids: np.ndarray
    block_offsets: np.ndarray
    block_row_counts: np.ndarray


@dataclass(frozen=True, slots=True)
class BlockCachePaths:
    """블록 캐시의 파일시스템 경로 집합."""

    store_root: Path
    chunk_root: Path
    lock_path: Path
    state_path: Path
    cached_path: Path
    chunk_ids_path: Path
    block_offsets_path: Path
    block_row_counts_path: Path


def resolve_cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    """캐시 루트 디렉토리 결정. None이면 ``~/.cache/cr-train`` 사용."""
    if cache_dir is not None:
        root = Path(cache_dir)
    else:
        root = Path.home() / ".cache" / "cr-train"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_dataset_seed(dataset_seed: int | None) -> int:
    return DEFAULT_DATASET_SEED if dataset_seed is None else int(dataset_seed)


def resolve_block_cache_paths(
    source_root: Path,
    split: str,
    dataset_seed: int,
    shuffle_buffer_size: int,
    *,
    predecoded: bool = False,
) -> BlockCachePaths:
    dirname = f"dataset-seed={dataset_seed}-shuffle-buffer={shuffle_buffer_size}"
    if predecoded:
        dirname += "-predecoded"
    store_root = (
        source_root
        / "block_store"
        / split
        / dirname
    )
    store_root.mkdir(parents=True, exist_ok=True)
    chunk_root = store_root / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    return BlockCachePaths(
        store_root=store_root,
        chunk_root=chunk_root,
        lock_path=store_root / ".lock",
        state_path=store_root / "state.json",
        cached_path=store_root / "cached.npy",
        chunk_ids_path=store_root / "chunk_ids.npy",
        block_offsets_path=store_root / "block_offsets.npy",
        block_row_counts_path=store_root / "block_row_counts.npy",
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_numpy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp_path.replace(path)


def load_numpy(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        return np.load(handle, allow_pickle=False)


def remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        path.unlink()
        return
    for child in path.iterdir():
        if child.is_dir():
            remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def _is_stale_lock(lock_path: Path) -> bool:
    try:
        pid = int(lock_path.read_text().strip())
        os.kill(pid, 0)
        return False
    except (ValueError, ProcessLookupError):
        return True
    except (PermissionError, OSError):
        return False


@contextmanager
def file_lock(lock_path: Path):
    started_at = time.monotonic()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            if _is_stale_lock(lock_path):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() - started_at > LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(f"timed out waiting for cache lock: {lock_path}")
            time.sleep(LOCK_POLL_INTERVAL_SECONDS)

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def suppress_hf_datasets_progress_bars():
    was_enabled = is_progress_bar_enabled()
    if was_enabled:
        disable_progress_bars()
    try:
        yield
    finally:
        if was_enabled:
            enable_progress_bars()


def save_dataset_without_progress(dataset: Dataset, path: Path) -> None:
    with suppress_hf_datasets_progress_bars():
        dataset.save_to_disk(str(path))


def _chunk_path(chunk_root: Path, chunk_index: int) -> Path:
    return chunk_root / f"{chunk_index:08d}.arrow"


def _next_chunk_index(chunk_root: Path) -> int:
    existing = [
        int(path.stem)
        for path in chunk_root.iterdir()
        if path.is_file() and path.suffix == ".arrow" and path.stem.isdigit()
    ]
    return 0 if not existing else max(existing) + 1


def save_chunk(paths: BlockCachePaths, rows: list[dict[str, Any]], *, chunk_index: int | None = None) -> int:
    """청크를 Arrow IPC 파일로 저장. HF Datasets 우회하여 직접 pyarrow 사용."""
    if chunk_index is None:
        chunk_index = _next_chunk_index(paths.chunk_root)
    chunk_file = _chunk_path(paths.chunk_root, chunk_index)
    tmp_file = chunk_file.with_suffix(".tmp")
    remove_tree(tmp_file)
    table = pa.table({col: [row[col] for row in rows] for col in DATA_COLUMNS})
    with pa.OSFile(str(tmp_file), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table)
        writer.close()
    tmp_file.replace(chunk_file)
    return chunk_index


def as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise TypeError(f"unsupported binary payload type: {type(value)!r}")


def freeze_value(value: Any) -> Any:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [freeze_value(item) for item in value]
    return value


def freeze_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: freeze_value(value) for key, value in row.items() if key in DATA_COLUMNS}


def _decode_chw(buffer: bytes, shape: tuple[int, ...], *, src_dtype: np.dtype, expected_channels: int) -> np.ndarray:
    """바이너리를 float32 CHW 배열로 디코딩. predecode_row 내부 헬퍼."""
    raw = np.frombuffer(buffer, dtype=src_dtype).reshape(shape)
    if raw.shape[-1] == expected_channels and raw.shape[0] != expected_channels:
        raw = np.transpose(raw, (2, 0, 1))
    return np.ascontiguousarray(raw, dtype=np.float32)


def predecode_row(row: dict[str, Any]) -> dict[str, Any]:
    """frozen row에 DSen2-CR 스케일링을 적용하여 float32 CHW로 변환."""
    # DSen2-CR SAR: VV clip[-25,0]→[0,2], VH clip[-32.5,0]→[0,2]
    sar = _decode_chw(as_bytes(row["sar"]), tuple(row["sar_shape"]), src_dtype=np.float32, expected_channels=SAR_CHANNELS)
    np.clip(sar[0], -25.0, 0.0, out=sar[0])
    sar[0] += 25.0
    sar[0] *= 2.0 / 25.0
    np.clip(sar[1], -32.5, 0.0, out=sar[1])
    sar[1] += 32.5
    sar[1] *= 2.0 / 32.5
    # DSen2-CR Optical: clip[0,10000]→/2000→[0,5]
    cloudy = _decode_chw(as_bytes(row["cloudy"]), tuple(row["opt_shape"]), src_dtype=np.int16, expected_channels=OPTICAL_CHANNELS)
    np.clip(cloudy, 0, 10000, out=cloudy)
    cloudy /= 2000.0
    target = _decode_chw(as_bytes(row["target"]), tuple(row["opt_shape"]), src_dtype=np.int16, expected_channels=OPTICAL_CHANNELS)
    np.clip(target, 0, 10000, out=target)
    target /= 2000.0
    return {
        "sar": sar.tobytes(),
        "cloudy": cloudy.tobytes(),
        "target": target.tobytes(),
        "sar_shape": list(sar.shape),
        "opt_shape": list(cloudy.shape),
        "season": row.get("season", ""),
        "scene": row.get("scene", ""),
        "patch": row.get("patch", ""),
    }


def load_or_init_block_cache(
    paths: BlockCachePaths,
    *,
    dataset_seed: int,
    shuffle_buffer_size: int,
    total_rows: int,
) -> SplitBlockCache:
    total_blocks = int(math.ceil(total_rows / BLOCK_SIZE)) if total_rows > 0 else 0
    files = (
        paths.state_path,
        paths.cached_path,
        paths.chunk_ids_path,
        paths.block_offsets_path,
        paths.block_row_counts_path,
    )
    existing_count = sum(path.exists() for path in files)

    if existing_count == len(files):
        payload = read_json(paths.state_path)
        cache = SplitBlockCache(
            state=SplitBlockCacheState(
                dataset_seed=int(payload["dataset_seed"]),
                shuffle_buffer_size=int(payload["shuffle_buffer_size"]),
                block_size=int(payload["block_size"]),
                total_rows=int(payload["total_rows"]),
                total_blocks=int(payload["total_blocks"]),
                canonical_frontier_block=int(payload["canonical_frontier_block"]),
            ),
            cached=load_numpy(paths.cached_path),
            chunk_ids=load_numpy(paths.chunk_ids_path),
            block_offsets=load_numpy(paths.block_offsets_path),
            block_row_counts=load_numpy(paths.block_row_counts_path),
        )
    elif existing_count == 0:
        cache = SplitBlockCache(
            state=SplitBlockCacheState(
                dataset_seed=dataset_seed,
                shuffle_buffer_size=shuffle_buffer_size,
                block_size=BLOCK_SIZE,
                total_rows=total_rows,
                total_blocks=total_blocks,
                canonical_frontier_block=0,
            ),
            cached=np.zeros(total_blocks, dtype=np.bool_),
            chunk_ids=np.full(total_blocks, -1, dtype=np.int32),
            block_offsets=np.full(total_blocks, -1, dtype=np.int32),
            block_row_counts=np.zeros(total_blocks, dtype=np.int32),
        )
        write_block_cache(paths, cache)
    else:
        raise FileNotFoundError(f"block cache index is partially missing: {paths.store_root}")

    state = cache.state
    if (
        state.block_size != BLOCK_SIZE
        or state.dataset_seed != dataset_seed
        or state.shuffle_buffer_size != shuffle_buffer_size
        or state.total_rows != total_rows
        or state.total_blocks != total_blocks
        or len(cache.cached) != total_blocks
        or len(cache.chunk_ids) != total_blocks
        or len(cache.block_offsets) != total_blocks
        or len(cache.block_row_counts) != total_blocks
    ):
        raise ValueError(f"block cache index mismatch: {paths.store_root}")
    return cache


def write_block_cache(paths: BlockCachePaths, cache: SplitBlockCache) -> None:
    state = cache.state
    write_json_atomic(
        paths.state_path,
        {
            "dataset_seed": state.dataset_seed,
            "shuffle_buffer_size": state.shuffle_buffer_size,
            "block_size": state.block_size,
            "total_rows": state.total_rows,
            "total_blocks": state.total_blocks,
            "canonical_frontier_block": state.canonical_frontier_block,
        },
    )
    write_numpy_atomic(paths.cached_path, cache.cached)
    write_numpy_atomic(paths.chunk_ids_path, cache.chunk_ids)
    write_numpy_atomic(paths.block_offsets_path, cache.block_offsets)
    write_numpy_atomic(paths.block_row_counts_path, cache.block_row_counts)


def materialize_blocks(
    paths: BlockCachePaths,
    *,
    start_block: int,
    blocks: list[list[dict[str, Any]]],
    cache: SplitBlockCache,
    next_chunk_index: int | None = None,
) -> tuple[int, int]:
    """블록을 청크로 저장하고 캐시 인덱스 갱신. (저장된 블록 수, 다음 chunk index) 반환."""
    if not blocks:
        return 0, next_chunk_index if next_chunk_index is not None else _next_chunk_index(paths.chunk_root)

    chunk_index = save_chunk(paths, [row for block_rows in blocks for row in block_rows], chunk_index=next_chunk_index)
    row_offset = 0
    for block_offset, block_rows in enumerate(blocks):
        block_index = start_block + block_offset
        cache.cached[block_index] = True
        cache.chunk_ids[block_index] = chunk_index
        cache.block_offsets[block_index] = row_offset
        cache.block_row_counts[block_index] = len(block_rows)
        row_offset += len(block_rows)
    return len(blocks), chunk_index + 1


__all__ = [
    "BlockCachePaths",
    "SplitBlockCache",
    "SplitBlockCacheState",
    "as_bytes",
    "file_lock",
    "freeze_row",
    "load_or_init_block_cache",
    "load_numpy",
    "materialize_blocks",
    "read_json",
    "remove_tree",
    "resolve_block_cache_paths",
    "resolve_cache_root",
    "resolve_dataset_seed",
    "save_chunk",
    "save_dataset_without_progress",
    "suppress_hf_datasets_progress_bars",
    "write_block_cache",
    "write_json_atomic",
]
