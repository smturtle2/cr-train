from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


@dataclass(frozen=True, slots=True)
class MappedBlockPayload(Sequence[dict[str, Any]]):
    sar: np.ndarray
    cloudy: np.ndarray
    target: np.ndarray
    season: tuple[str, ...]
    scene: tuple[str, ...]
    patch: tuple[str, ...]
    sar_shape: tuple[tuple[int, int, int], ...]
    opt_shape: tuple[tuple[int, int, int], ...]

    def __len__(self) -> int:
        return int(self.sar.shape[0])

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]

        resolved_index = int(index)
        return {
            "sar": self.sar[resolved_index],
            "cloudy": self.cloudy[resolved_index],
            "target": self.target[resolved_index],
            "sar_shape": list(self.sar_shape[resolved_index]),
            "opt_shape": list(self.opt_shape[resolved_index]),
            "season": self.season[resolved_index],
            "scene": self.scene[resolved_index],
            "patch": self.patch[resolved_index],
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


_PAYLOAD_METADATA_FILENAME = "payload.json"
_SAR_PAYLOAD_FILENAME = "sar.npy"
_CLOUDY_PAYLOAD_FILENAME = "cloudy.npy"
_TARGET_PAYLOAD_FILENAME = "target.npy"
_BLOCK_PAYLOAD_FILENAMES = (
    _PAYLOAD_METADATA_FILENAME,
    _SAR_PAYLOAD_FILENAME,
    _CLOUDY_PAYLOAD_FILENAME,
    _TARGET_PAYLOAD_FILENAME,
)


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
    return paths.block_root / cache_key


def block_metadata_path(paths: BlockCachePaths, cache_key: str) -> Path:
    return paths.metadata_root / f"{cache_key}.json"


def block_lock_path(paths: BlockCachePaths, cache_key: str) -> Path:
    return paths.lock_root / f"{cache_key}.lock"


def _tmp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _payload_metadata_path(payload_path: Path) -> Path:
    return payload_path / _PAYLOAD_METADATA_FILENAME


def _payload_file_path(payload_path: Path, filename: str) -> Path:
    return payload_path / filename


def _payload_files_exist(payload_path: Path) -> bool:
    return payload_path.is_dir() and all(_payload_file_path(payload_path, filename).is_file() for filename in _BLOCK_PAYLOAD_FILENAMES)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return sum(_path_size(child) for child in path.iterdir())


def _as_shape(value: Any, *, field: str) -> tuple[int, int, int]:
    del field
    shape = tuple(int(dim) for dim in value)
    return shape  # type: ignore[return-value]


def _decode_image_array(buffer: Any, *, shape: tuple[int, int, int], dtype: np.dtype[Any], field: str) -> np.ndarray:
    del field
    return np.frombuffer(as_bytes(buffer), dtype=dtype).reshape(shape)


def _build_block_payload(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    row_count = len(rows)
    sar_shape = _as_shape(rows[0]["sar_shape"], field="sar_shape")
    opt_shape = _as_shape(rows[0]["opt_shape"], field="opt_shape")
    sar = np.empty((row_count, *sar_shape), dtype=np.float32)
    cloudy = np.empty((row_count, *opt_shape), dtype=np.int16)
    target = np.empty((row_count, *opt_shape), dtype=np.int16)
    season: list[str] = []
    scene: list[str] = []
    patch: list[str] = []
    sar_shapes: list[list[int]] = []
    opt_shapes: list[list[int]] = []

    for index, row in enumerate(rows):
        row_sar_shape = _as_shape(row["sar_shape"], field="sar_shape")
        row_opt_shape = _as_shape(row["opt_shape"], field="opt_shape")
        sar[index] = _decode_image_array(row["sar"], shape=row_sar_shape, dtype=np.dtype(np.float32), field="sar")
        cloudy[index] = _decode_image_array(
            row["cloudy"],
            shape=row_opt_shape,
            dtype=np.dtype(np.int16),
            field="cloudy",
        )
        target[index] = _decode_image_array(
            row["target"],
            shape=row_opt_shape,
            dtype=np.dtype(np.int16),
            field="target",
        )
        season.append(str(row.get("season", "")))
        scene.append(str(row.get("scene", "")))
        patch.append(str(row.get("patch", "")))
        sar_shapes.append(list(row_sar_shape))
        opt_shapes.append(list(row_opt_shape))

    payload_metadata = {
        "row_count": row_count,
        "season": season,
        "scene": scene,
        "patch": patch,
        "sar_shape": sar_shapes,
        "opt_shape": opt_shapes,
    }
    return sar, cloudy, target, payload_metadata


def clear_block_cache_entry(paths: BlockCachePaths, cache_key: str, *, keep_lock: bool = False) -> None:
    payload_path = block_data_path(paths, cache_key)
    metadata_path = block_metadata_path(paths, cache_key)
    lock_path = block_lock_path(paths, cache_key)

    remove_tree(payload_path)
    remove_tree(_tmp_path(payload_path))
    remove_tree(metadata_path)
    remove_tree(_tmp_path(metadata_path))
    if not keep_lock:
        remove_tree(lock_path)
        remove_tree(_tmp_path(lock_path))


def block_is_cached(paths: BlockCachePaths, cache_key: str) -> bool:
    return _payload_files_exist(block_data_path(paths, cache_key)) and block_metadata_path(paths, cache_key).is_file()


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

    payload_tmp = _tmp_path(payload_path)
    metadata_tmp = _tmp_path(metadata_path)
    remove_tree(payload_tmp)
    remove_tree(metadata_tmp)
    remove_tree(payload_path)

    sar, cloudy, target, payload_metadata = _build_block_payload(rows)
    payload_tmp.mkdir(parents=True, exist_ok=True)
    np.save(_payload_file_path(payload_tmp, _SAR_PAYLOAD_FILENAME), sar, allow_pickle=False)
    np.save(_payload_file_path(payload_tmp, _CLOUDY_PAYLOAD_FILENAME), cloudy, allow_pickle=False)
    np.save(_payload_file_path(payload_tmp, _TARGET_PAYLOAD_FILENAME), target, allow_pickle=False)
    _payload_metadata_path(payload_tmp).write_text(
        json.dumps(payload_metadata, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    metadata_tmp.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")

    payload_tmp.replace(payload_path)
    metadata_tmp.replace(metadata_path)
    return SaveBlockResult(
        payload_bytes=_path_size(payload_path),
        metadata_bytes=metadata_path.stat().st_size,
    )


def _load_payload_shapes(payload_metadata: dict[str, Any], *, field: str) -> tuple[tuple[int, int, int], ...]:
    return tuple(_as_shape(shape, field=field) for shape in payload_metadata[field])


def _load_payload_strings(payload_metadata: dict[str, Any], *, field: str) -> tuple[str, ...]:
    return tuple(str(item) for item in payload_metadata[field])


def load_block(paths: BlockCachePaths, cache_key: str) -> MappedBlockPayload:
    path = block_data_path(paths, cache_key)
    payload_metadata = read_json(_payload_metadata_path(path))
    sar = np.load(_payload_file_path(path, _SAR_PAYLOAD_FILENAME), mmap_mode="r")
    cloudy = np.load(_payload_file_path(path, _CLOUDY_PAYLOAD_FILENAME), mmap_mode="r")
    target = np.load(_payload_file_path(path, _TARGET_PAYLOAD_FILENAME), mmap_mode="r")

    return MappedBlockPayload(
        sar=sar,
        cloudy=cloudy,
        target=target,
        season=_load_payload_strings(payload_metadata, field="season"),
        scene=_load_payload_strings(payload_metadata, field="scene"),
        patch=_load_payload_strings(payload_metadata, field="patch"),
        sar_shape=_load_payload_shapes(payload_metadata, field="sar_shape"),
        opt_shape=_load_payload_shapes(payload_metadata, field="opt_shape"),
    )


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
    "MappedBlockPayload",
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
