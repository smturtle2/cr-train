from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import LOCK_POLL_INTERVAL_SECONDS, LOCK_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class BlockCachePaths:
    store_root: Path
    block_root: Path
    metadata_root: Path
    lock_root: Path


@dataclass(frozen=True, slots=True)
class SaveBlockResult:
    payload_bytes: int
    metadata_bytes: int

    @property
    def written_bytes(self) -> int:
        return self.payload_bytes + self.metadata_bytes


def resolve_cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    """Resolve the cache root directory. Defaults to ``~/.cache/cr-train``."""
    return Path(cache_dir) if cache_dir is not None else (Path.home() / ".cache" / "cr-train")


def resolve_block_cache_paths(source_root: Path, split: str) -> BlockCachePaths:
    store_root = source_root / "block_store" / split
    store_root.mkdir(parents=True, exist_ok=True)
    block_root = store_root / "blocks"
    block_root.mkdir(parents=True, exist_ok=True)
    metadata_root = store_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    lock_root = store_root / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    return BlockCachePaths(
        store_root=store_root,
        block_root=block_root,
        metadata_root=metadata_root,
        lock_root=lock_root,
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def block_data_path(paths: BlockCachePaths, cache_key: str) -> Path:
    return paths.block_root / f"{cache_key}.pt"


def block_metadata_path(paths: BlockCachePaths, cache_key: str) -> Path:
    return paths.metadata_root / f"{cache_key}.json"


def block_lock_path(paths: BlockCachePaths, cache_key: str) -> Path:
    return paths.lock_root / f"{cache_key}.lock"


def clear_block_cache_entry(paths: BlockCachePaths, cache_key: str, *, keep_lock: bool = False) -> None:
    payload_path = block_data_path(paths, cache_key)
    metadata_path = block_metadata_path(paths, cache_key)
    lock_path = block_lock_path(paths, cache_key)

    remove_tree(payload_path)
    remove_tree(payload_path.with_suffix(payload_path.suffix + ".tmp"))
    remove_tree(metadata_path)
    remove_tree(metadata_path.with_suffix(metadata_path.suffix + ".tmp"))
    if not keep_lock:
        remove_tree(lock_path)
        remove_tree(lock_path.with_suffix(lock_path.suffix + ".tmp"))


def block_is_cached(paths: BlockCachePaths, cache_key: str) -> bool:
    return block_data_path(paths, cache_key).exists() and block_metadata_path(paths, cache_key).exists()


def load_block_metadata(paths: BlockCachePaths, cache_key: str) -> dict[str, Any] | None:
    path = block_metadata_path(paths, cache_key)
    if not path.exists():
        return None
    return read_json(path)


def save_block(
    paths: BlockCachePaths,
    *,
    cache_key: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> SaveBlockResult:
    payload_path = block_data_path(paths, cache_key)
    metadata_path = block_metadata_path(paths, cache_key)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    payload_tmp = payload_path.with_suffix(payload_path.suffix + ".tmp")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    remove_tree(payload_tmp)
    remove_tree(metadata_tmp)

    torch.save(rows, payload_tmp)
    metadata_tmp.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")

    payload_tmp.replace(payload_path)
    metadata_tmp.replace(metadata_path)
    return SaveBlockResult(
        payload_bytes=payload_path.stat().st_size,
        metadata_bytes=metadata_path.stat().st_size,
    )


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_block(paths: BlockCachePaths, cache_key: str) -> list[dict[str, Any]]:
    path = block_data_path(paths, cache_key)
    if not path.exists():
        raise FileNotFoundError(f"cached block is missing: {path}")
    payload = _torch_load(path)
    if not isinstance(payload, list):
        raise TypeError(f"cached block payload must be a list, got {type(payload)!r}")
    return payload


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


__all__ = [
    "BlockCachePaths",
    "SaveBlockResult",
    "as_bytes",
    "block_data_path",
    "block_is_cached",
    "block_lock_path",
    "block_metadata_path",
    "clear_block_cache_entry",
    "file_lock",
    "freeze_row",
    "load_block",
    "load_block_metadata",
    "read_json",
    "remove_tree",
    "resolve_block_cache_paths",
    "resolve_cache_root",
    "save_block",
    "write_json_atomic",
]
