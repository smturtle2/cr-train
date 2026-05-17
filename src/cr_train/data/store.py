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

from .constants import BLOCK_SIZE, LOCK_POLL_INTERVAL_SECONDS, LOCK_TIMEOUT_SECONDS


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
    lock_path.parent.mkdir(parents=True, exist_ok=True)
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
                raise TimeoutError(f"timed out waiting for data lock: {lock_path}")
            time.sleep(LOCK_POLL_INTERVAL_SECONDS)

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _shape_tuple(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"shape must be a list or tuple, got {type(value)!r}")
    shape = tuple(int(dim) for dim in value)
    if len(shape) != 3:
        raise ValueError(f"expected a 3D tensor shape, got {shape!r}")
    return shape  # type: ignore[return-value]


def as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).tobytes()
    raise TypeError(f"expected bytes-like value, got {type(value)!r}")


def _decode_array(buffer: Any, shape_value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    shape = _shape_tuple(shape_value)
    if isinstance(buffer, np.ndarray):
        array = np.asarray(buffer, dtype=dtype)
        if array.shape != shape:
            array = array.reshape(shape)
        return np.ascontiguousarray(array)
    return np.frombuffer(as_bytes(buffer), dtype=dtype).reshape(shape)


def _build_block_payload(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not rows:
        raise ValueError("cannot build an empty block")
    if len(rows) > BLOCK_SIZE:
        raise ValueError(f"block row count exceeds BLOCK_SIZE={BLOCK_SIZE}: {len(rows)}")

    sar_arrays: list[np.ndarray] = []
    cloudy_arrays: list[np.ndarray] = []
    target_arrays: list[np.ndarray] = []
    sar_shapes: list[list[int]] = []
    opt_shapes: list[list[int]] = []
    seasons: list[str] = []
    scenes: list[str] = []
    patches: list[str] = []
    for row in rows:
        sar_shape = list(_shape_tuple(row["sar_shape"]))
        opt_shape = list(_shape_tuple(row["opt_shape"]))
        sar_arrays.append(_decode_array(row["sar"], sar_shape, dtype=np.dtype("float32")))
        cloudy_arrays.append(_decode_array(row["cloudy"], opt_shape, dtype=np.dtype("int16")))
        target_arrays.append(_decode_array(row["target"], opt_shape, dtype=np.dtype("int16")))
        sar_shapes.append(sar_shape)
        opt_shapes.append(opt_shape)
        seasons.append(str(row.get("season", "")))
        scenes.append(str(row.get("scene", "")))
        patches.append(str(row.get("patch", "")))

    payload_metadata = {
        "row_count": len(rows),
        "season": seasons,
        "scene": scenes,
        "patch": patches,
        "sar_shape": sar_shapes,
        "opt_shape": opt_shapes,
    }
    return (
        np.stack(sar_arrays),
        np.stack(cloudy_arrays),
        np.stack(target_arrays),
        payload_metadata,
    )


def build_mapped_block_payload(
    *,
    payload_metadata: dict[str, Any],
    sar: np.ndarray,
    cloudy: np.ndarray,
    target: np.ndarray,
) -> MappedBlockPayload:
    return MappedBlockPayload(
        sar=sar,
        cloudy=cloudy,
        target=target,
        season=tuple(str(value) for value in payload_metadata["season"]),
        scene=tuple(str(value) for value in payload_metadata["scene"]),
        patch=tuple(str(value) for value in payload_metadata["patch"]),
        sar_shape=tuple(tuple(int(dim) for dim in shape) for shape in payload_metadata["sar_shape"]),
        opt_shape=tuple(tuple(int(dim) for dim in shape) for shape in payload_metadata["opt_shape"]),
    )


__all__ = [
    "MappedBlockPayload",
    "SaveBlockResult",
    "as_bytes",
    "build_mapped_block_payload",
    "file_lock",
    "read_json",
    "remove_tree",
    "write_json_atomic",
]
