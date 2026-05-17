from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zstandard as zstd

from .constants import CRPACK_BLOCK_SIZE, CRPACK_LAYOUT_VERSION
from .store import (
    MappedBlockPayload,
    SaveBlockResult,
    _build_block_payload,
    build_mapped_block_payload,
    remove_tree,
)

CRPACK_MAGIC = b"CRPACK15\n"
CRPACK_CODEC = "zstd"
CRPACK_LEVEL = 1
CRPACK_EXTENSION = ".crpack"
_HEADER_LEN = struct.Struct("<Q")
CRPACK_HEADER_PREFIX_SIZE = len(CRPACK_MAGIC) + _HEADER_LEN.size


@dataclass(frozen=True, slots=True)
class V15BlockLocation:
    source_root: Path
    split: str


@dataclass(frozen=True, slots=True)
class V15LocalBlockReader:
    source_root: Path
    split: str

    def load_block(self, cache_key: str) -> MappedBlockPayload:
        return load_v15_block(self.source_root, self.split, cache_key)

    def block_is_ready(self, cache_key: str) -> bool:
        return v15_block_is_cached(self.source_root, self.split, cache_key)

    def load_block_metadata(self, cache_key: str) -> dict[str, Any] | None:
        return load_v15_block_metadata(self.source_root, self.split, cache_key)

    def load_block_metadata_many(self, cache_keys: tuple[str, ...]) -> dict[str, dict[str, Any] | None]:
        return {cache_key: self.load_block_metadata(cache_key) for cache_key in cache_keys}


def resolve_v15_block_store_root(source_root: Path, split: str) -> Path:
    path = source_root / "block_store" / split
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_v15_block_root(source_root: Path, split: str) -> Path:
    path = resolve_v15_block_store_root(source_root, split) / "blocks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_v15_lock_root(source_root: Path, split: str) -> Path:
    path = resolve_v15_block_store_root(source_root, split) / "locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def v15_block_path(source_root: Path, split: str, cache_key: str) -> Path:
    return resolve_v15_block_root(source_root, split) / f"{cache_key}{CRPACK_EXTENSION}"


def v15_block_lock_path(source_root: Path, split: str, cache_key: str) -> Path:
    return resolve_v15_lock_root(source_root, split) / f"{cache_key}.lock"


def _v15_tmp_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def clear_v15_block(source_root: Path, split: str, cache_key: str) -> None:
    path = v15_block_path(source_root, split, cache_key)
    remove_tree(path)
    remove_tree(_v15_tmp_path(path))


def v15_block_is_cached(source_root: Path, split: str, cache_key: str) -> bool:
    return v15_block_path(source_root, split, cache_key).is_file()


def _byte_shuffle(array: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(array)
    itemsize = int(contiguous.dtype.itemsize)
    if itemsize <= 1:
        return contiguous.tobytes(order="C")
    raw = contiguous.view(np.uint8).reshape(-1, itemsize)
    return raw.T.copy().tobytes(order="C")


def _byte_unshuffle(payload: bytes, *, dtype: np.dtype[Any], shape: tuple[int, ...]) -> bytes:
    itemsize = int(dtype.itemsize)
    if itemsize <= 1:
        return payload
    shuffled = np.frombuffer(payload, dtype=np.uint8).reshape(itemsize, -1)
    return shuffled.T.copy().reshape(-1).tobytes(order="C")


def _array_header(array: np.ndarray, *, offset: int, shuffled_size: int) -> dict[str, Any]:
    return {
        "dtype": array.dtype.str,
        "shape": [int(dim) for dim in array.shape],
        "offset": int(offset),
        "nbytes": int(shuffled_size),
        "raw_nbytes": int(array.nbytes),
        "shuffle": "byte",
    }


def pack_v15_block(
    *,
    payload_metadata: dict[str, Any],
    metadata: dict[str, Any],
    sar: np.ndarray,
    cloudy: np.ndarray,
    target: np.ndarray,
) -> bytes:
    row_count = int(payload_metadata["row_count"])
    if row_count <= 0 or row_count > CRPACK_BLOCK_SIZE:
        raise ValueError(f"row_count must be between 1 and {CRPACK_BLOCK_SIZE}, got {row_count}")

    payload_parts: list[bytes] = []
    arrays: dict[str, dict[str, Any]] = {}
    offset = 0
    raw_bytes = 0
    for name, array in (
        ("sar", np.ascontiguousarray(sar)),
        ("cloudy", np.ascontiguousarray(cloudy)),
        ("target", np.ascontiguousarray(target)),
    ):
        shuffled = _byte_shuffle(array)
        payload_parts.append(shuffled)
        arrays[name] = _array_header(array, offset=offset, shuffled_size=len(shuffled))
        offset += len(shuffled)
        raw_bytes += int(array.nbytes)

    compressed = zstd.ZstdCompressor(level=CRPACK_LEVEL).compress(b"".join(payload_parts))
    header = {
        "version": CRPACK_LAYOUT_VERSION,
        "codec": CRPACK_CODEC,
        "level": CRPACK_LEVEL,
        "cache_block_size": CRPACK_BLOCK_SIZE,
        "row_count": row_count,
        "raw_bytes": raw_bytes,
        "compressed_bytes": len(compressed),
        "arrays": arrays,
        "payload_metadata": payload_metadata,
        "metadata": metadata,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return CRPACK_MAGIC + _HEADER_LEN.pack(len(header_bytes)) + header_bytes + compressed


def _read_header_from_bytes(blob: bytes) -> tuple[dict[str, Any], int]:
    if not blob.startswith(CRPACK_MAGIC):
        raise ValueError("not a cr-train v15 crpack payload")
    header_len_offset = len(CRPACK_MAGIC)
    header_len = _HEADER_LEN.unpack(blob[header_len_offset : header_len_offset + _HEADER_LEN.size])[0]
    header_start = header_len_offset + _HEADER_LEN.size
    header_end = header_start + int(header_len)
    header = json.loads(blob[header_start:header_end].decode("utf-8"))
    return header, header_end


def read_v15_header_length(prefix: bytes) -> int:
    if len(prefix) < CRPACK_HEADER_PREFIX_SIZE or not prefix.startswith(CRPACK_MAGIC):
        raise ValueError("not a cr-train v15 crpack header")
    return int(_HEADER_LEN.unpack(prefix[len(CRPACK_MAGIC) : CRPACK_HEADER_PREFIX_SIZE])[0])


def read_v15_header_from_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        magic = handle.read(len(CRPACK_MAGIC))
        if magic != CRPACK_MAGIC:
            raise ValueError(f"not a cr-train v15 crpack payload: {path}")
        header_len = _HEADER_LEN.unpack(handle.read(_HEADER_LEN.size))[0]
        return json.loads(handle.read(int(header_len)).decode("utf-8"))


def unpack_v15_block_bytes(blob: bytes) -> tuple[MappedBlockPayload, dict[str, Any]]:
    header, payload_offset = _read_header_from_bytes(blob)
    if int(header.get("version", 0)) != CRPACK_LAYOUT_VERSION:
        raise ValueError(f"unsupported crpack version: {header.get('version')!r}")
    if header.get("codec") != CRPACK_CODEC:
        raise ValueError(f"unsupported crpack codec: {header.get('codec')!r}")

    decompressed = zstd.ZstdDecompressor().decompress(blob[payload_offset:])
    loaded_arrays: dict[str, np.ndarray] = {}
    for name in ("sar", "cloudy", "target"):
        info = header["arrays"][name]
        dtype = np.dtype(str(info["dtype"]))
        shape = tuple(int(dim) for dim in info["shape"])
        start = int(info["offset"])
        end = start + int(info["nbytes"])
        raw = _byte_unshuffle(decompressed[start:end], dtype=dtype, shape=shape)
        loaded_arrays[name] = np.frombuffer(raw, dtype=dtype).copy().reshape(shape)

    return (
        build_mapped_block_payload(
            payload_metadata=header["payload_metadata"],
            sar=loaded_arrays["sar"],
            cloudy=loaded_arrays["cloudy"],
            target=loaded_arrays["target"],
        ),
        dict(header["metadata"]),
    )


def write_v15_block_arrays(
    *,
    source_root: Path,
    split: str,
    cache_key: str,
    payload_metadata: dict[str, Any],
    metadata: dict[str, Any],
    sar: np.ndarray,
    cloudy: np.ndarray,
    target: np.ndarray,
) -> SaveBlockResult:
    path = v15_block_path(source_root, split, cache_key)
    tmp_path = _v15_tmp_path(path)
    remove_tree(tmp_path)
    blob = pack_v15_block(
        payload_metadata=payload_metadata,
        metadata=metadata,
        sar=sar,
        cloudy=cloudy,
        target=target,
    )
    tmp_path.write_bytes(blob)
    tmp_path.replace(path)
    return SaveBlockResult(payload_bytes=path.stat().st_size, metadata_bytes=0)


def write_v15_block(
    *,
    source_root: Path,
    split: str,
    cache_key: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> SaveBlockResult:
    sar, cloudy, target, payload_metadata = _build_block_payload(rows)
    return write_v15_block_arrays(
        source_root=source_root,
        split=split,
        cache_key=cache_key,
        payload_metadata=payload_metadata,
        metadata=metadata,
        sar=sar,
        cloudy=cloudy,
        target=target,
    )


def write_v15_block_bytes(
    *,
    source_root: Path,
    split: str,
    cache_key: str,
    blob: bytes,
) -> SaveBlockResult:
    path = v15_block_path(source_root, split, cache_key)
    tmp_path = _v15_tmp_path(path)
    remove_tree(tmp_path)
    tmp_path.write_bytes(blob)
    tmp_path.replace(path)
    return SaveBlockResult(payload_bytes=path.stat().st_size, metadata_bytes=0)


def load_v15_block(source_root: Path, split: str, cache_key: str) -> MappedBlockPayload:
    path = v15_block_path(source_root, split, cache_key)
    try:
        payload, _metadata = unpack_v15_block_bytes(path.read_bytes())
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, zstd.ZstdError) as exc:
        clear_v15_block(source_root, split, cache_key)
        raise RuntimeError(
            f"v15 cached block payload is unreadable for {cache_key}; cache entry was cleared"
        ) from exc
    return payload


def load_v15_block_metadata(source_root: Path, split: str, cache_key: str) -> dict[str, Any] | None:
    path = v15_block_path(source_root, split, cache_key)
    if not path.exists():
        return None
    try:
        header = read_v15_header_from_file(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        clear_v15_block(source_root, split, cache_key)
        raise RuntimeError(f"v15 cached block metadata is unreadable for {cache_key}") from exc
    return dict(header["metadata"])


__all__ = [
    "CRPACK_EXTENSION",
    "CRPACK_HEADER_PREFIX_SIZE",
    "CRPACK_MAGIC",
    "V15BlockLocation",
    "V15LocalBlockReader",
    "clear_v15_block",
    "load_v15_block",
    "load_v15_block_metadata",
    "pack_v15_block",
    "read_v15_header_from_file",
    "read_v15_header_length",
    "resolve_v15_block_root",
    "resolve_v15_block_store_root",
    "resolve_v15_lock_root",
    "unpack_v15_block_bytes",
    "v15_block_is_cached",
    "v15_block_lock_path",
    "v15_block_path",
    "write_v15_block",
    "write_v15_block_arrays",
    "write_v15_block_bytes",
]
