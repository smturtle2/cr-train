from __future__ import annotations

import io
import json
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np

from .constants import CACHE_LAYOUT_VERSION
from .store import (
    BlockCachePaths,
    MappedBlockPayload,
    build_mapped_block_payload,
    block_data_path,
    block_is_cached,
    block_metadata_path,
    clear_block_cache_entry,
    load_block,
    parse_completed_marker_name,
    remove_tree,
    resolve_block_cache_paths,
    write_completed_block_marker,
)


CacheSource = Literal["local", "B2"]

_B2_ENV_VARS = ("B2_BUCKET", "B2_ENDPOINT", "B2_KEY_ID", "B2_APP_KEY")
_B2_CACHE_PREFIX = "cache"
_PAYLOAD_METADATA_FILENAME = "payload.json"
_SAR_PAYLOAD_FILENAME = "sar.npy"
_CLOUDY_PAYLOAD_FILENAME = "cloudy.npy"
_TARGET_PAYLOAD_FILENAME = "target.npy"
_B2_DOWNLOAD_WORKERS = 16
_B2_DOWNLOAD_CHUNK_SIZE = 16 * 1024 * 1024
_B2_READ_ATTEMPTS = 4
_B2_RETRY_BASE_DELAY_SECONDS = 0.25
_B2_STAGING_POLL_SECONDS = 0.05
_B2_STAGING_DEFAULT_MAX_BLOCKS = 20
_B2_STAGING_TARGET_INFLIGHT_BYTES = 2 * 1024 * 1024 * 1024


class BlockReader(Protocol):
    def load_block(self, cache_key: str) -> MappedBlockPayload:
        ...


@dataclass(frozen=True, slots=True)
class LocalBlockReader:
    cache_paths: BlockCachePaths

    def load_block(self, cache_key: str) -> MappedBlockPayload:
        return load_block(self.cache_paths, cache_key)


@dataclass(frozen=True, slots=True)
class B2SourceLocation:
    source_prefix: str
    descriptor: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _B2RangeTask:
    cache_key: str
    object_key: str
    filename: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class _B2ObjectDownloadState:
    filename: str
    object_key: str
    ranges: tuple[tuple[int, int], ...]
    chunks: dict[int, bytes]

    @property
    def complete(self) -> bool:
        return len(self.chunks) == len(self.ranges)


@dataclass(slots=True)
class _B2BlockDownloadState:
    cache_key: str
    payload_metadata_bytes: bytes
    metadata_bytes: bytes
    objects: dict[str, _B2ObjectDownloadState]
    pending_ranges: deque[_B2RangeTask]

    @property
    def complete(self) -> bool:
        return all(state.complete for state in self.objects.values())


def normalize_cache_src(value: str) -> CacheSource:
    normalized = value.strip().lower()
    if normalized == "local":
        return "local"
    if normalized == "b2":
        return "B2"
    supported = "local, B2"
    raise ValueError(f"cache_src must be one of {supported}")


def b2_download_worker_count() -> int:
    return _B2_DOWNLOAD_WORKERS


def _join_key(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def _normalize_endpoint_url(value: str) -> str:
    if value.startswith(("http://", "https://")):
        return value
    return f"https://{value}"


def _read_stream_body(body: Any) -> bytes:
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    return body.read()


def _read_object_body(response: dict[str, Any]) -> bytes:
    body = response["Body"]
    try:
        return _read_stream_body(body)
    finally:
        close = getattr(body, "close", None)
        if close is not None:
            close()


def _missing_client_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    return str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}


def _retryable_client_error(exc: Exception) -> bool:
    if type(exc).__name__ in {
        "ConnectionClosedError",
        "EndpointConnectionError",
        "ReadTimeoutError",
        "ResponseStreamingError",
    }:
        return True
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    if not isinstance(error, dict):
        return False
    return str(error.get("Code")) in {
        "500",
        "502",
        "503",
        "504",
        "InternalError",
        "RequestTimeout",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
    }


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(_B2_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))


def _payload_object_filenames() -> tuple[str, str, str]:
    return (_SAR_PAYLOAD_FILENAME, _CLOUDY_PAYLOAD_FILENAME, _TARGET_PAYLOAD_FILENAME)


def _write_staged_block(
    *,
    paths: BlockCachePaths,
    cache_key: str,
    payload_metadata_bytes: bytes,
    metadata_bytes: bytes,
    payloads_by_filename: dict[str, bytes],
) -> None:
    payload_metadata = json.loads(payload_metadata_bytes.decode("utf-8"))
    payload_path = block_data_path(paths, cache_key)
    metadata_path = block_metadata_path(paths, cache_key)
    payload_tmp = payload_path.with_suffix(payload_path.suffix + ".tmp")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    clear_block_cache_entry(paths, cache_key)
    remove_tree(payload_tmp)
    remove_tree(metadata_tmp)
    payload_tmp.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    (payload_tmp / _PAYLOAD_METADATA_FILENAME).write_bytes(payload_metadata_bytes)
    for filename in _payload_object_filenames():
        (payload_tmp / filename).write_bytes(payloads_by_filename[filename])
    metadata_tmp.write_bytes(metadata_bytes)
    payload_tmp.replace(payload_path)
    metadata_tmp.replace(metadata_path)
    write_completed_block_marker(paths, cache_key, row_count=int(payload_metadata["row_count"]))


def _new_s3_client(*, endpoint_url: str, key_id: str, app_key: str):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(
            max_pool_connections=_B2_DOWNLOAD_WORKERS,
            retries={"max_attempts": _B2_READ_ATTEMPTS, "mode": "standard"},
        ),
    )


class B2CacheRepository:
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str,
        key_id: str,
        app_key: str,
        prefix: str | os.PathLike[str] = _B2_CACHE_PREFIX,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = _normalize_endpoint_url(endpoint_url)
        self.key_id = key_id
        self.app_key = app_key
        self.prefix = os.fspath(prefix).strip("/")
        self._client = client
        self._client_provided = client is not None

    @classmethod
    def from_env(cls, *, prefix: str | os.PathLike[str] | None = None) -> B2CacheRepository:
        missing = [name for name in _B2_ENV_VARS if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"B2 cache requires environment variables: {', '.join(missing)}")
        return cls(
            bucket=os.environ["B2_BUCKET"],
            endpoint_url=os.environ["B2_ENDPOINT"],
            key_id=os.environ["B2_KEY_ID"],
            app_key=os.environ["B2_APP_KEY"],
            prefix=_B2_CACHE_PREFIX if prefix is None else prefix,
        )

    def _client_for_request(self):
        if self._client is None:
            self._client = _new_s3_client(
                endpoint_url=self.endpoint_url,
                key_id=self.key_id,
                app_key=self.app_key,
            )
        return self._client

    def _read_bytes(self, key: str) -> bytes:
        for attempt in range(_B2_READ_ATTEMPTS):
            try:
                response = self._client_for_request().get_object(Bucket=self.bucket, Key=key)
                return _read_object_body(response)
            except Exception as exc:
                if _missing_client_error(exc):
                    raise FileNotFoundError(f"B2 cache object is missing: b2://{self.bucket}/{key}") from exc
                if attempt < _B2_READ_ATTEMPTS - 1 and _retryable_client_error(exc):
                    _sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"failed to read B2 cache object: b2://{self.bucket}/{key}") from exc
        raise RuntimeError(f"failed to read B2 cache object: b2://{self.bucket}/{key}")

    def _read_json(self, key: str) -> dict[str, Any]:
        try:
            return json.loads(self._read_bytes(key).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"B2 cache object is not valid JSON: b2://{self.bucket}/{key}") from exc

    def _list_objects(self, *, prefix: str, delimiter: str | None = None):
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Prefix": prefix,
        }
        if delimiter is not None:
            kwargs["Delimiter"] = delimiter

        while True:
            response = self._client_for_request().list_objects_v2(**kwargs)
            yield response
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
            if token is None:
                break
            kwargs["ContinuationToken"] = token

    def find_source(self, *, dataset_name: str, revision: str | None) -> B2SourceLocation:
        layout_prefix = _join_key(self.prefix, f"layout-v{CACHE_LAYOUT_VERSION}") + "/"
        source_prefixes: list[str] = []
        for response in self._list_objects(prefix=layout_prefix, delimiter="/"):
            source_prefixes.extend(
                str(item["Prefix"]).rstrip("/")
                for item in response.get("CommonPrefixes", [])
                if "Prefix" in item
            )

        for source_prefix in source_prefixes:
            try:
                descriptor = self._read_json(_join_key(source_prefix, "source.json"))
            except FileNotFoundError:
                continue
            if descriptor.get("dataset_name") == dataset_name and descriptor.get("revision") == revision:
                return B2SourceLocation(source_prefix=source_prefix, descriptor=descriptor)

        raise FileNotFoundError(
            "B2 cache source metadata was not found for "
            f"dataset_name={dataset_name!r}, revision={revision!r}, prefix=b2://{self.bucket}/{layout_prefix}"
        )

    def load_split_catalog(self, *, source: B2SourceLocation, split: str) -> dict[str, Any]:
        split_sizes = source.descriptor.get("split_sizes", {})
        if split not in split_sizes:
            raise KeyError(f"split {split!r} does not exist in B2 source descriptor")
        return self._read_json(_join_key(source.source_prefix, "catalogs", f"{split}.json"))

    def block_reader(self, *, source: B2SourceLocation, split: str) -> B2BlockReader:
        return B2BlockReader(
            bucket=self.bucket,
            endpoint_url=self.endpoint_url,
            key_id=self.key_id,
            app_key=self.app_key,
            source_prefix=source.source_prefix,
            split=split,
            client=self._client,
        )

    def staged_block_reader(
        self,
        *,
        source: B2SourceLocation,
        split: str,
        staging_root: Path,
        max_staged_blocks: int = _B2_STAGING_DEFAULT_MAX_BLOCKS,
    ) -> B2StagedBlockReader:
        return B2StagedBlockReader(
            bucket=self.bucket,
            endpoint_url=self.endpoint_url,
            key_id=self.key_id,
            app_key=self.app_key,
            source_prefix=source.source_prefix,
            split=split,
            staging_root=staging_root,
            max_staged_blocks=max_staged_blocks,
            client=self._client if self._client_provided else None,
        )


class B2BlockReader:
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str,
        key_id: str,
        app_key: str,
        source_prefix: str,
        split: str,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.key_id = key_id
        self.app_key = app_key
        self.source_prefix = source_prefix.rstrip("/")
        self.split = split
        self._client = client

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_client"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _client_for_request(self):
        if self._client is None:
            self._client = _new_s3_client(
                endpoint_url=self.endpoint_url,
                key_id=self.key_id,
                app_key=self.app_key,
            )
        return self._client

    def _read_bytes(self, key: str) -> bytes:
        for attempt in range(_B2_READ_ATTEMPTS):
            try:
                response = self._client_for_request().get_object(Bucket=self.bucket, Key=key)
                return _read_object_body(response)
            except Exception as exc:
                if _missing_client_error(exc):
                    raise FileNotFoundError(f"B2 cache object is missing: b2://{self.bucket}/{key}") from exc
                if attempt < _B2_READ_ATTEMPTS - 1 and _retryable_client_error(exc):
                    _sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"failed to read B2 cache object: b2://{self.bucket}/{key}") from exc
        raise RuntimeError(f"failed to read B2 cache object: b2://{self.bucket}/{key}")

    def _read_object_range(self, key: str, *, start: int, end: int) -> tuple[int, bytes]:
        for attempt in range(_B2_READ_ATTEMPTS):
            try:
                response = self._client_for_request().get_object(
                    Bucket=self.bucket,
                    Key=key,
                    Range=f"bytes={start}-{end - 1}",
                )
                return start, _read_object_body(response)
            except Exception as exc:
                if _missing_client_error(exc):
                    raise FileNotFoundError(f"B2 cache object is missing: b2://{self.bucket}/{key}") from exc
                if attempt < _B2_READ_ATTEMPTS - 1 and _retryable_client_error(exc):
                    _sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"failed to read B2 cache object range: b2://{self.bucket}/{key}") from exc
        raise RuntimeError(f"failed to read B2 cache object range: b2://{self.bucket}/{key}")

    def _object_size(self, key: str) -> int:
        for attempt in range(_B2_READ_ATTEMPTS):
            try:
                response = self._client_for_request().head_object(Bucket=self.bucket, Key=key)
                return int(response["ContentLength"])
            except Exception as exc:
                if _missing_client_error(exc):
                    raise FileNotFoundError(f"B2 cache object is missing: b2://{self.bucket}/{key}") from exc
                if attempt < _B2_READ_ATTEMPTS - 1 and _retryable_client_error(exc):
                    _sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"failed to read B2 cache object metadata: b2://{self.bucket}/{key}") from exc
        raise RuntimeError(f"failed to read B2 cache object metadata: b2://{self.bucket}/{key}")

    def _read_large_bytes(self, key: str) -> bytes:
        return self._read_large_bytes_many((key,))[key]

    def _read_large_bytes_many(self, keys: tuple[str, ...]) -> dict[str, bytes]:
        object_sizes = {key: self._object_size(key) for key in keys}
        if len(keys) == 1:
            key = keys[0]
            object_size = object_sizes[key]
            if object_size > _B2_DOWNLOAD_CHUNK_SIZE:
                return {key: self._read_ranged_bytes(key, object_size=object_size)}
            return {key: self._read_bytes(key)}

        ranges_by_key: dict[str, list[tuple[int, int]]] = {}
        payloads: dict[str, bytes] = {}
        futures: dict[Any, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=_B2_DOWNLOAD_WORKERS) as pool:
            for key, object_size in object_sizes.items():
                if object_size <= _B2_DOWNLOAD_CHUNK_SIZE:
                    futures[pool.submit(self._read_bytes, key)] = ("full", key)
                    continue
                ranges = [
                    (start, min(start + _B2_DOWNLOAD_CHUNK_SIZE, object_size))
                    for start in range(0, object_size, _B2_DOWNLOAD_CHUNK_SIZE)
                ]
                ranges_by_key[key] = ranges
                for start, end in ranges:
                    futures[pool.submit(self._read_object_range, key, start=start, end=end)] = ("range", key)

            chunks_by_key: dict[str, dict[int, bytes]] = {
                key: {} for key in ranges_by_key
            }
            for future in as_completed(futures):
                kind, key = futures[future]
                if kind == "full":
                    payloads[key] = future.result()
                    continue
                start, payload = future.result()
                chunks_by_key[key][start] = payload

        for key, ranges in ranges_by_key.items():
            chunks = chunks_by_key[key]
            payloads[key] = b"".join(chunks[start] for start, _end in ranges)
        return payloads

    def _read_ranged_bytes(self, key: str, *, object_size: int) -> bytes:
        ranges = [
            (start, min(start + _B2_DOWNLOAD_CHUNK_SIZE, object_size))
            for start in range(0, object_size, _B2_DOWNLOAD_CHUNK_SIZE)
        ]
        chunks: dict[int, bytes] = {}
        with ThreadPoolExecutor(max_workers=_B2_DOWNLOAD_WORKERS) as pool:
            futures = [
                pool.submit(self._read_object_range, key, start=start, end=end)
                for start, end in ranges
            ]
            for future in as_completed(futures):
                start, payload = future.result()
                chunks[start] = payload
        return b"".join(chunks[start] for start, _end in ranges)

    def _read_json(self, key: str) -> dict[str, Any]:
        try:
            return json.loads(self._read_bytes(key).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"B2 cache object is not valid JSON: b2://{self.bucket}/{key}") from exc

    def _list_objects(self, *, prefix: str):
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Prefix": prefix,
        }
        while True:
            response = self._client_for_request().list_objects_v2(**kwargs)
            yield response
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
            if token is None:
                break
            kwargs["ContinuationToken"] = token

    def _block_store_key(self, *parts: str) -> str:
        return _join_key(self.source_prefix, "block_store", self.split, *parts)

    def _block_payload_keys(self, cache_key: str) -> dict[str, str]:
        return {
            _SAR_PAYLOAD_FILENAME: self._block_store_key(
                "blocks", cache_key, _SAR_PAYLOAD_FILENAME
            ),
            _CLOUDY_PAYLOAD_FILENAME: self._block_store_key(
                "blocks", cache_key, _CLOUDY_PAYLOAD_FILENAME
            ),
            _TARGET_PAYLOAD_FILENAME: self._block_store_key(
                "blocks", cache_key, _TARGET_PAYLOAD_FILENAME
            ),
        }

    def _block_payload_metadata_key(self, cache_key: str) -> str:
        return self._block_store_key("blocks", cache_key, _PAYLOAD_METADATA_FILENAME)

    def _block_metadata_key(self, cache_key: str) -> str:
        return self._block_store_key("metadata", f"{cache_key}.json")

    def load_completed_block_index(self) -> dict[str, int]:
        completed: dict[str, int] = {}
        prefix = self._block_store_key("completed") + "/"
        for response in self._list_objects(prefix=prefix):
            for item in response.get("Contents", []):
                key = str(item.get("Key", ""))
                parsed = parse_completed_marker_name(key.rsplit("/", 1)[-1])
                if parsed is None:
                    continue
                cache_key, row_count = parsed
                completed[cache_key] = row_count
        return completed

    def load_block_metadata(self, cache_key: str) -> dict[str, Any] | None:
        key = self._block_metadata_key(cache_key)
        try:
            return self._read_json(key)
        except FileNotFoundError:
            return None

    def load_block_metadata_many(self, cache_keys: tuple[str, ...]) -> dict[str, dict[str, Any] | None]:
        if not cache_keys:
            return {}
        with ThreadPoolExecutor(max_workers=min(_B2_DOWNLOAD_WORKERS, len(cache_keys))) as pool:
            futures = {
                pool.submit(self.load_block_metadata, cache_key): cache_key
                for cache_key in cache_keys
            }
            return {cache_key: future.result() for future, cache_key in futures.items()}

    def stage_block(self, cache_key: str, paths: BlockCachePaths) -> None:
        payload_keys = self._block_payload_keys(cache_key)
        payload_metadata_bytes = self._read_bytes(self._block_payload_metadata_key(cache_key))
        metadata_bytes = self._read_bytes(self._block_metadata_key(cache_key))
        payloads = self._read_large_bytes_many(tuple(payload_keys.values()))
        _write_staged_block(
            paths=paths,
            cache_key=cache_key,
            payload_metadata_bytes=payload_metadata_bytes,
            metadata_bytes=metadata_bytes,
            payloads_by_filename={
                filename: payloads[object_key]
                for filename, object_key in payload_keys.items()
            },
        )

    def load_block(self, cache_key: str) -> MappedBlockPayload:
        try:
            payload_metadata = self._read_json(self._block_payload_metadata_key(cache_key))
            payload_keys = self._block_payload_keys(cache_key)
            payloads = self._read_large_bytes_many(tuple(payload_keys.values()))
            sar = self._load_npy_bytes(payloads[payload_keys[_SAR_PAYLOAD_FILENAME]])
            cloudy = self._load_npy_bytes(payloads[payload_keys[_CLOUDY_PAYLOAD_FILENAME]])
            target = self._load_npy_bytes(payloads[payload_keys[_TARGET_PAYLOAD_FILENAME]])
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError(f"B2 cached block payload is unreadable for {cache_key}") from exc

        return build_mapped_block_payload(
            payload_metadata=payload_metadata,
            sar=sar,
            cloudy=cloudy,
            target=target,
        )

    def _load_npy(self, key: str) -> np.ndarray:
        with io.BytesIO(self._read_large_bytes(key)) as buffer:
            return cast(np.ndarray, np.load(buffer, allow_pickle=False))

    def _load_npy_bytes(self, payload: bytes) -> np.ndarray:
        with io.BytesIO(payload) as buffer:
            return cast(np.ndarray, np.load(buffer, allow_pickle=False))


def _staging_error_path(paths: BlockCachePaths) -> Path:
    return paths.store_root / "staging.error"


def _write_staging_error(paths: BlockCachePaths, exc: BaseException) -> None:
    _staging_error_path(paths).write_text(str(exc), encoding="utf-8")


def _raise_if_staging_error(paths: BlockCachePaths) -> None:
    error_path = _staging_error_path(paths)
    if error_path.exists():
        message = error_path.read_text(encoding="utf-8").strip()
        raise RuntimeError(f"B2 staging downloader failed: {message}")


def _count_staged_blocks(paths: BlockCachePaths, cache_keys: tuple[str, ...]) -> int:
    return sum(1 for cache_key in cache_keys if block_is_cached(paths, cache_key))


def _object_ranges(object_size: int) -> tuple[tuple[int, int], ...]:
    if object_size <= 0:
        raise RuntimeError("B2 cache object is empty")
    return tuple(
        (start, min(start + _B2_DOWNLOAD_CHUNK_SIZE, object_size))
        for start in range(0, object_size, _B2_DOWNLOAD_CHUNK_SIZE)
    )


def _build_b2_block_download_state(
    *, reader: B2BlockReader, cache_key: str
) -> _B2BlockDownloadState:
    payload_keys = reader._block_payload_keys(cache_key)
    object_states: dict[str, _B2ObjectDownloadState] = {}
    max_range_count = 0
    for filename, object_key in payload_keys.items():
        ranges = _object_ranges(reader._object_size(object_key))
        object_states[filename] = _B2ObjectDownloadState(
            filename=filename,
            object_key=object_key,
            ranges=ranges,
            chunks={},
        )
        max_range_count = max(max_range_count, len(ranges))

    pending_ranges: deque[_B2RangeTask] = deque()
    for range_index in range(max_range_count):
        for filename in _payload_object_filenames():
            object_state = object_states[filename]
            if range_index >= len(object_state.ranges):
                continue
            start, end = object_state.ranges[range_index]
            pending_ranges.append(
                _B2RangeTask(
                    cache_key=cache_key,
                    object_key=object_state.object_key,
                    filename=filename,
                    start=start,
                    end=end,
                )
            )

    return _B2BlockDownloadState(
        cache_key=cache_key,
        payload_metadata_bytes=reader._read_bytes(reader._block_payload_metadata_key(cache_key)),
        metadata_bytes=reader._read_bytes(reader._block_metadata_key(cache_key)),
        objects=object_states,
        pending_ranges=pending_ranges,
    )


def _write_b2_download_state(paths: BlockCachePaths, state: _B2BlockDownloadState) -> None:
    payloads_by_filename: dict[str, bytes] = {}
    for filename, object_state in state.objects.items():
        payloads_by_filename[filename] = b"".join(
            object_state.chunks[start] for start, _end in object_state.ranges
        )
    _write_staged_block(
        paths=paths,
        cache_key=state.cache_key,
        payload_metadata_bytes=state.payload_metadata_bytes,
        metadata_bytes=state.metadata_bytes,
        payloads_by_filename=payloads_by_filename,
    )


def _schedule_b2_range_tasks(
    *,
    pool: ThreadPoolExecutor,
    reader: B2BlockReader,
    active_blocks: dict[str, _B2BlockDownloadState],
    futures: dict[Future[tuple[int, bytes]], _B2RangeTask],
    queued_bytes: int,
) -> int:
    while queued_bytes < _B2_STAGING_TARGET_INFLIGHT_BYTES:
        submitted = False
        for state in list(active_blocks.values()):
            if queued_bytes >= _B2_STAGING_TARGET_INFLIGHT_BYTES:
                break
            if not state.pending_ranges:
                continue
            task = state.pending_ranges.popleft()
            future = pool.submit(
                reader._read_object_range,
                task.object_key,
                start=task.start,
                end=task.end,
            )
            futures[future] = task
            queued_bytes += task.size
            submitted = True
        if not submitted:
            break
    return queued_bytes


def _run_b2_staging_downloader(
    *,
    bucket: str,
    endpoint_url: str,
    key_id: str,
    app_key: str,
    source_prefix: str,
    split: str,
    staging_source_root: str,
    cache_keys: tuple[str, ...],
    max_staged_blocks: int,
) -> None:
    paths = resolve_block_cache_paths(Path(staging_source_root), split)
    reader = B2BlockReader(
        bucket=bucket,
        endpoint_url=endpoint_url,
        key_id=key_id,
        app_key=app_key,
        source_prefix=source_prefix,
        split=split,
    )
    try:
        pending_cache_keys: deque[str] = deque(cache_keys)
        active_blocks: dict[str, _B2BlockDownloadState] = {}
        queued_bytes = 0
        futures: dict[Future[tuple[int, bytes]], _B2RangeTask] = {}
        completed_any_block = False
        with ThreadPoolExecutor(max_workers=_B2_DOWNLOAD_WORKERS) as pool:
            while pending_cache_keys or active_blocks or futures:
                active_block_limit = max_staged_blocks if completed_any_block else 1
                while (
                    pending_cache_keys
                    and len(active_blocks) + _count_staged_blocks(paths, cache_keys)
                    < active_block_limit
                ):
                    cache_key = pending_cache_keys.popleft()
                    if block_is_cached(paths, cache_key):
                        continue
                    active_blocks[cache_key] = _build_b2_block_download_state(
                        reader=reader,
                        cache_key=cache_key,
                    )

                queued_bytes = _schedule_b2_range_tasks(
                    pool=pool,
                    reader=reader,
                    active_blocks=active_blocks,
                    futures=futures,
                    queued_bytes=queued_bytes,
                )

                if not futures:
                    time.sleep(_B2_STAGING_POLL_SECONDS)
                    continue

                done, _pending = wait(
                    futures,
                    timeout=_B2_STAGING_POLL_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    task = futures.pop(future)
                    queued_bytes -= task.size
                    start, payload = future.result()
                    state = active_blocks[task.cache_key]
                    object_state = state.objects[task.filename]
                    object_state.chunks[start] = payload
                    if state.complete:
                        _write_b2_download_state(paths, state)
                        del active_blocks[task.cache_key]
                        completed_any_block = True
    except BaseException as exc:
        _write_staging_error(paths, exc)
        raise


class B2StagedBlockReader:
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str,
        key_id: str,
        app_key: str,
        source_prefix: str,
        split: str,
        staging_root: Path,
        max_staged_blocks: int = _B2_STAGING_DEFAULT_MAX_BLOCKS,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.key_id = key_id
        self.app_key = app_key
        self.source_prefix = source_prefix.rstrip("/")
        self.split = split
        self.staging_root = Path(staging_root)
        self.max_staged_blocks = max(1, int(max_staged_blocks))
        source_key = str(abs(hash((self.source_prefix, os.getpid(), id(self)))))
        self.staging_source_root = self.staging_root / source_key
        self.cache_paths = resolve_block_cache_paths(self.staging_source_root, split)
        self._client = client
        self._remote_reader = B2BlockReader(
            bucket=bucket,
            endpoint_url=endpoint_url,
            key_id=key_id,
            app_key=app_key,
            source_prefix=source_prefix,
            split=split,
            client=client,
        )
        self._process: mp.Process | None = None
        self._process_owner_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_client"] = None
        state["_remote_reader"] = None
        state["_process"] = None
        state["_process_owner_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def load_completed_block_index(self) -> dict[str, int]:
        if self._remote_reader is None:
            self._remote_reader = B2BlockReader(
                bucket=self.bucket,
                endpoint_url=self.endpoint_url,
                key_id=self.key_id,
                app_key=self.app_key,
                source_prefix=self.source_prefix,
                split=self.split,
                client=self._client,
            )
        return self._remote_reader.load_completed_block_index()

    def load_block_metadata(self, cache_key: str) -> dict[str, Any] | None:
        if self._remote_reader is None:
            self._remote_reader = B2BlockReader(
                bucket=self.bucket,
                endpoint_url=self.endpoint_url,
                key_id=self.key_id,
                app_key=self.app_key,
                source_prefix=self.source_prefix,
                split=self.split,
                client=self._client,
            )
        return self._remote_reader.load_block_metadata(cache_key)

    def load_block_metadata_many(self, cache_keys: tuple[str, ...]) -> dict[str, dict[str, Any] | None]:
        if self._remote_reader is None:
            self._remote_reader = B2BlockReader(
                bucket=self.bucket,
                endpoint_url=self.endpoint_url,
                key_id=self.key_id,
                app_key=self.app_key,
                source_prefix=self.source_prefix,
                split=self.split,
                client=self._client,
            )
        return self._remote_reader.load_block_metadata_many(cache_keys)

    def prepare_blocks(self, blocks: tuple[dict[str, Any], ...]) -> None:
        cache_keys = tuple(str(block["cache_key"]) for block in blocks)
        remove_tree(self.staging_source_root)
        self.cache_paths = resolve_block_cache_paths(self.staging_source_root, self.split)
        if self._client is not None:
            if self._remote_reader is None:
                self._remote_reader = B2BlockReader(
                    bucket=self.bucket,
                    endpoint_url=self.endpoint_url,
                    key_id=self.key_id,
                    app_key=self.app_key,
                    source_prefix=self.source_prefix,
                    split=self.split,
                    client=self._client,
                )
            for cache_key in cache_keys:
                self._remote_reader.stage_block(cache_key, self.cache_paths)
            return
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._process = mp.Process(
            target=_run_b2_staging_downloader,
            kwargs={
                "bucket": self.bucket,
                "endpoint_url": self.endpoint_url,
                "key_id": self.key_id,
                "app_key": self.app_key,
                "source_prefix": self.source_prefix,
                "split": self.split,
                "staging_source_root": os.fspath(self.staging_source_root),
                "cache_keys": cache_keys,
                "max_staged_blocks": self.max_staged_blocks,
            },
            daemon=True,
        )
        self._process.start()
        self._process_owner_pid = os.getpid()

    def load_block(self, cache_key: str) -> MappedBlockPayload:
        while not block_is_cached(self.cache_paths, cache_key):
            _raise_if_staging_error(self.cache_paths)
            process = self._process if self._process_owner_pid == os.getpid() else None
            if process is not None and not process.is_alive() and process.exitcode not in (None, 0):
                raise RuntimeError(f"B2 staging downloader exited with code {process.exitcode}")
            time.sleep(_B2_STAGING_POLL_SECONDS)
        return load_block(self.cache_paths, cache_key)

    def release_block(self, cache_key: str) -> None:
        clear_block_cache_entry(self.cache_paths, cache_key)


def resolve_b2_cache_repository(
    *, prefix: str | os.PathLike[str] | None = None
) -> B2CacheRepository:
    return B2CacheRepository.from_env(prefix=prefix)


__all__ = [
    "B2BlockReader",
    "B2CacheRepository",
    "B2SourceLocation",
    "B2StagedBlockReader",
    "BlockReader",
    "CacheSource",
    "LocalBlockReader",
    "b2_download_worker_count",
    "normalize_cache_src",
    "resolve_b2_cache_repository",
]
