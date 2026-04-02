from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc

from .constants import LOCK_POLL_INTERVAL_SECONDS, LOCK_TIMEOUT_SECONDS


@dataclass(slots=True)
class SplitRowCacheState:
    """Persistent state for the split-wide row cache."""

    total_rows: int
    total_cached_rows: int
    next_chunk_index: int


@dataclass(slots=True)
class SplitRowCache:
    """In-memory row cache index keyed by global row id."""

    state: SplitRowCacheState
    cached: np.ndarray
    chunk_ids: np.ndarray
    row_offsets: np.ndarray


@dataclass(frozen=True, slots=True)
class RowCachePaths:
    store_root: Path
    chunk_root: Path
    lock_path: Path
    state_path: Path
    cached_path: Path
    chunk_ids_path: Path
    row_offsets_path: Path


def resolve_cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    """Resolve the cache root directory. Defaults to ``~/.cache/cr-train``."""
    root = Path(cache_dir) if cache_dir is not None else (Path.home() / ".cache" / "cr-train")
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_row_cache_paths(source_root: Path, split: str) -> RowCachePaths:
    store_root = source_root / "row_store" / split
    store_root.mkdir(parents=True, exist_ok=True)
    chunk_root = store_root / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    return RowCachePaths(
        store_root=store_root,
        chunk_root=chunk_root,
        lock_path=store_root / ".lock",
        state_path=store_root / "state.json",
        cached_path=store_root / "cached_rows.npy",
        chunk_ids_path=store_root / "chunk_ids.npy",
        row_offsets_path=store_root / "row_offsets.npy",
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

def _chunk_path(chunk_root: Path, chunk_index: int) -> Path:
    return chunk_root / f"{chunk_index:08d}.arrow"


def _existing_chunk_ids(chunk_root: Path) -> set[int]:
    return {
        int(path.stem)
        for path in chunk_root.iterdir()
        if path.is_file() and path.suffix == ".arrow" and path.stem.isdigit()
    }


def _next_chunk_index(chunk_root: Path) -> int:
    existing = _existing_chunk_ids(chunk_root)
    return 0 if not existing else (max(existing) + 1)


def save_chunk(paths: RowCachePaths, rows: list[dict[str, Any]], *, chunk_index: int | None = None) -> int:
    """Save rows as one Arrow IPC chunk and return the chunk index."""
    if chunk_index is None:
        chunk_index = _next_chunk_index(paths.chunk_root)
    chunk_file = _chunk_path(paths.chunk_root, chunk_index)
    tmp_file = chunk_file.with_suffix(".tmp")
    remove_tree(tmp_file)
    if rows:
        schema_data = {col: [row[col] for row in rows] for col in rows[0]}
    else:
        schema_data = {}
    table = pa.table(schema_data)
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
    return {key: freeze_value(value) for key, value in row.items()}


def _init_empty_row_cache(total_rows: int) -> SplitRowCache:
    return SplitRowCache(
        state=SplitRowCacheState(total_rows=total_rows, total_cached_rows=0, next_chunk_index=0),
        cached=np.zeros(total_rows, dtype=np.bool_),
        chunk_ids=np.full(total_rows, -1, dtype=np.int32),
        row_offsets=np.full(total_rows, -1, dtype=np.int32),
    )


def write_row_cache(paths: RowCachePaths, cache: SplitRowCache) -> None:
    state = cache.state
    write_json_atomic(
        paths.state_path,
        {
            "total_rows": state.total_rows,
            "total_cached_rows": state.total_cached_rows,
            "next_chunk_index": state.next_chunk_index,
        },
    )
    write_numpy_atomic(paths.cached_path, cache.cached)
    write_numpy_atomic(paths.chunk_ids_path, cache.chunk_ids)
    write_numpy_atomic(paths.row_offsets_path, cache.row_offsets)


def _heal_missing_chunks(paths: RowCachePaths, cache: SplitRowCache) -> bool:
    valid_chunk_ids = _existing_chunk_ids(paths.chunk_root)
    if not valid_chunk_ids and cache.state.total_cached_rows == 0:
        return False

    invalid_mask = np.logical_and(cache.cached, ~np.isin(cache.chunk_ids, list(valid_chunk_ids)))
    if not np.any(invalid_mask):
        return False

    cache.cached[invalid_mask] = False
    cache.chunk_ids[invalid_mask] = -1
    cache.row_offsets[invalid_mask] = -1
    cache.state.total_cached_rows = int(np.count_nonzero(cache.cached))
    return True


def load_or_init_row_cache(paths: RowCachePaths, *, total_rows: int) -> SplitRowCache:
    files = (
        paths.state_path,
        paths.cached_path,
        paths.chunk_ids_path,
        paths.row_offsets_path,
    )
    existing_count = sum(path.exists() for path in files)

    if existing_count == len(files):
        payload = read_json(paths.state_path)
        cache = SplitRowCache(
            state=SplitRowCacheState(
                total_rows=int(payload["total_rows"]),
                total_cached_rows=int(payload["total_cached_rows"]),
                next_chunk_index=int(payload["next_chunk_index"]),
            ),
            cached=load_numpy(paths.cached_path),
            chunk_ids=load_numpy(paths.chunk_ids_path),
            row_offsets=load_numpy(paths.row_offsets_path),
        )
    elif existing_count == 0:
        cache = _init_empty_row_cache(total_rows)
        write_row_cache(paths, cache)
        return cache
    else:
        cache = _init_empty_row_cache(total_rows)
        write_row_cache(paths, cache)
        return cache

    if (
        cache.state.total_rows != total_rows
        or cache.cached.shape != (total_rows,)
        or cache.chunk_ids.shape != (total_rows,)
        or cache.row_offsets.shape != (total_rows,)
    ):
        cache = _init_empty_row_cache(total_rows)
        write_row_cache(paths, cache)
        return cache

    cached_count = int(np.count_nonzero(cache.cached))
    if cache.state.total_cached_rows != cached_count:
        cache.state.total_cached_rows = cached_count
    if cache.state.next_chunk_index < 0:
        cache.state.next_chunk_index = _next_chunk_index(paths.chunk_root)
    else:
        cache.state.next_chunk_index = max(cache.state.next_chunk_index, _next_chunk_index(paths.chunk_root))

    healed = _heal_missing_chunks(paths, cache)
    if healed:
        cache.state.next_chunk_index = _next_chunk_index(paths.chunk_root)
        write_row_cache(paths, cache)
    return cache


def materialize_rows(
    paths: RowCachePaths,
    *,
    row_entries: list[tuple[int, dict[str, Any]]],
    cache: SplitRowCache,
) -> int:
    """Append fetched rows to a new chunk and update the row index."""
    if not row_entries:
        return 0

    unique_entries: list[tuple[int, dict[str, Any]]] = []
    seen_row_ids: set[int] = set()
    for row_id, row in row_entries:
        if row_id in seen_row_ids:
            continue
        seen_row_ids.add(row_id)
        unique_entries.append((row_id, row))
    if not unique_entries:
        return 0

    chunk_index = save_chunk(
        paths,
        [row for _, row in unique_entries],
        chunk_index=cache.state.next_chunk_index,
    )
    inserted = 0
    for row_offset, (row_id, _) in enumerate(unique_entries):
        if row_id < 0 or row_id >= cache.state.total_rows:
            raise IndexError(f"row id {row_id} is out of range for cache")
        if not bool(cache.cached[row_id]):
            inserted += 1
        cache.cached[row_id] = True
        cache.chunk_ids[row_id] = chunk_index
        cache.row_offsets[row_id] = row_offset
    cache.state.total_cached_rows = int(np.count_nonzero(cache.cached))
    cache.state.next_chunk_index = chunk_index + 1
    return inserted


__all__ = [
    "RowCachePaths",
    "SplitRowCache",
    "SplitRowCacheState",
    "as_bytes",
    "file_lock",
    "freeze_row",
    "load_numpy",
    "load_or_init_row_cache",
    "materialize_rows",
    "read_json",
    "remove_tree",
    "resolve_cache_root",
    "resolve_row_cache_paths",
    "save_chunk",
    "write_json_atomic",
    "write_row_cache",
]
