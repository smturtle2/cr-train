from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import zstandard as zstd

from .constants import CACHE_LAYOUT_VERSION
from .store import (
    MappedBlockPayload,
    remove_tree,
)
from .v15 import (
    CRPACK_EXTENSION,
    CRPACK_HEADER_PREFIX_SIZE,
    V15LocalBlockReader,
    clear_v15_block,
    read_v15_header_length,
    load_v15_block,
    unpack_v15_block_bytes,
    v15_block_is_cached,
    write_v15_block_bytes,
)


CacheSource = Literal["local", "B2"]

_B2_ENV_VARS = ("B2_BUCKET", "B2_ENDPOINT", "B2_KEY_ID", "B2_APP_KEY")
_B2_CACHE_PREFIX = "cache"
_B2_DOWNLOAD_WORKERS = 24
_B2_READ_ATTEMPTS = 4
_B2_RETRY_BASE_DELAY_SECONDS = 0.25
_B2_STAGING_POLL_SECONDS = 0.05
_B2_STAGING_DEFAULT_MAX_BLOCKS = 80


class BlockReader(Protocol):
    def load_block(self, cache_key: str) -> MappedBlockPayload:
        ...


LocalBlockReader = V15LocalBlockReader


@dataclass(frozen=True, slots=True)
class B2SourceLocation:
    source_prefix: str
    descriptor: dict[str, Any]


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


def resolve_b2_download_workers(download_workers: int | None = None) -> int:
    if download_workers is None:
        return _B2_DOWNLOAD_WORKERS
    resolved = int(download_workers)
    if resolved <= 0:
        raise ValueError("b2_download_workers must be greater than zero")
    return resolved


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


def _write_staged_crpack(
    *,
    staging_source_root: Path,
    split: str,
    cache_key: str,
    payload: bytes,
) -> None:
    write_v15_block_bytes(
        source_root=staging_source_root,
        split=split,
        cache_key=cache_key,
        blob=payload,
    )


def _new_s3_client(*, endpoint_url: str, key_id: str, app_key: str, download_workers: int):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(
            max_pool_connections=download_workers,
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
        download_workers: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = _normalize_endpoint_url(endpoint_url)
        self.key_id = key_id
        self.app_key = app_key
        self.prefix = os.fspath(prefix).strip("/")
        self.download_workers = resolve_b2_download_workers(download_workers)
        self._client = client
        self._client_provided = client is not None

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str | os.PathLike[str] | None = None,
        download_workers: int | None = None,
    ) -> B2CacheRepository:
        missing = [name for name in _B2_ENV_VARS if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"B2 cache requires environment variables: {', '.join(missing)}")
        return cls(
            bucket=os.environ["B2_BUCKET"],
            endpoint_url=os.environ["B2_ENDPOINT"],
            key_id=os.environ["B2_KEY_ID"],
            app_key=os.environ["B2_APP_KEY"],
            prefix=_B2_CACHE_PREFIX if prefix is None else prefix,
            download_workers=download_workers,
        )

    def _client_for_request(self):
        if self._client is None:
            self._client = _new_s3_client(
                endpoint_url=self.endpoint_url,
                key_id=self.key_id,
                app_key=self.app_key,
                download_workers=self.download_workers,
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
            download_workers=self.download_workers,
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
            download_workers=self.download_workers,
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
        download_workers: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.key_id = key_id
        self.app_key = app_key
        self.source_prefix = source_prefix.rstrip("/")
        self.split = split
        self.download_workers = resolve_b2_download_workers(download_workers)
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
                download_workers=self.download_workers,
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

    def _block_crpack_key(self, cache_key: str) -> str:
        return self._block_store_key("blocks", f"{cache_key}{CRPACK_EXTENSION}")

    def load_completed_block_index(self) -> dict[str, int]:
        completed: dict[str, int] = {}
        prefix = self._block_store_key("blocks") + "/"
        for response in self._list_objects(prefix=prefix):
            for item in response.get("Contents", []):
                key = str(item.get("Key", ""))
                name = key.rsplit("/", 1)[-1]
                if not name.endswith(CRPACK_EXTENSION):
                    continue
                cache_key = name[: -len(CRPACK_EXTENSION)]
                metadata = self.load_block_metadata(cache_key)
                if metadata is not None:
                    completed[cache_key] = int(metadata["row_count"])
        return completed

    def _read_crpack_header(self, key: str) -> dict[str, Any]:
        _start, prefix = self._read_object_range(
            key,
            start=0,
            end=CRPACK_HEADER_PREFIX_SIZE,
        )
        header_len = read_v15_header_length(prefix)
        _start, header_bytes = self._read_object_range(
            key,
            start=CRPACK_HEADER_PREFIX_SIZE,
            end=CRPACK_HEADER_PREFIX_SIZE + header_len,
        )
        try:
            return json.loads(header_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"B2 crpack header is not valid JSON: b2://{self.bucket}/{key}") from exc

    def load_block_metadata(self, cache_key: str) -> dict[str, Any] | None:
        key = self._block_crpack_key(cache_key)
        try:
            header = self._read_crpack_header(key)
            return dict(header["metadata"])
        except FileNotFoundError:
            return None

    def load_block_metadata_many(self, cache_keys: tuple[str, ...]) -> dict[str, dict[str, Any] | None]:
        if not cache_keys:
            return {}
        with ThreadPoolExecutor(max_workers=min(self.download_workers, len(cache_keys))) as pool:
            futures = {
                pool.submit(self.load_block_metadata, cache_key): cache_key
                for cache_key in cache_keys
            }
            return {cache_key: future.result() for future, cache_key in futures.items()}

    def stage_block(self, cache_key: str, *, staging_source_root: Path, split: str) -> None:
        payload = self._read_bytes(self._block_crpack_key(cache_key))
        _write_staged_crpack(
            staging_source_root=staging_source_root,
            split=split,
            cache_key=cache_key,
            payload=payload,
        )

    def load_block(self, cache_key: str) -> MappedBlockPayload:
        try:
            payload, _metadata = unpack_v15_block_bytes(self._read_bytes(self._block_crpack_key(cache_key)))
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, zstd.ZstdError) as exc:
            raise RuntimeError(f"B2 cached block payload is unreadable for {cache_key}") from exc
        return payload


def _staging_error_path(staging_source_root: Path, split: str) -> Path:
    return staging_source_root / "block_store" / split / "staging.error"


def _write_staging_error(staging_source_root: Path, split: str, exc: BaseException) -> None:
    error_path = _staging_error_path(staging_source_root, split)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(str(exc), encoding="utf-8")


def _raise_if_staging_error(staging_source_root: Path, split: str) -> None:
    error_path = _staging_error_path(staging_source_root, split)
    if error_path.exists():
        message = error_path.read_text(encoding="utf-8").strip()
        raise RuntimeError(f"B2 staging downloader failed: {message}")


def _count_staged_blocks(staging_source_root: Path, split: str, cache_keys: tuple[str, ...]) -> int:
    return sum(1 for cache_key in cache_keys if v15_block_is_cached(staging_source_root, split, cache_key))


def _download_and_stage_crpack(
    *,
    reader: B2BlockReader,
    staging_source_root: Path,
    split: str,
    cache_key: str,
) -> str:
    if v15_block_is_cached(staging_source_root, split, cache_key):
        return cache_key
    payload = reader._read_bytes(reader._block_crpack_key(cache_key))
    _write_staged_crpack(
        staging_source_root=staging_source_root,
        split=split,
        cache_key=cache_key,
        payload=payload,
    )
    return cache_key


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
    download_workers: int,
) -> None:
    staging_root_path = Path(staging_source_root)
    reader = B2BlockReader(
        bucket=bucket,
        endpoint_url=endpoint_url,
        key_id=key_id,
        app_key=app_key,
        source_prefix=source_prefix,
        split=split,
        download_workers=download_workers,
    )
    try:
        pending_cache_keys: deque[str] = deque(cache_keys)
        if pending_cache_keys:
            first_cache_key = pending_cache_keys.popleft()
            _download_and_stage_crpack(
                reader=reader,
                staging_source_root=staging_root_path,
                split=split,
                cache_key=first_cache_key,
            )

        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            futures: dict[Future[str], str] = {}
            while pending_cache_keys or futures:
                while (
                    pending_cache_keys
                    and len(futures) < download_workers
                    and _count_staged_blocks(staging_root_path, split, cache_keys) + len(futures)
                    < max_staged_blocks
                ):
                    cache_key = pending_cache_keys.popleft()
                    if v15_block_is_cached(staging_root_path, split, cache_key):
                        continue
                    futures[
                        pool.submit(
                            _download_and_stage_crpack,
                            reader=reader,
                            staging_source_root=staging_root_path,
                            split=split,
                            cache_key=cache_key,
                        )
                    ] = cache_key

                if not futures:
                    time.sleep(_B2_STAGING_POLL_SECONDS)
                    continue

                done, _pending = wait(
                    futures,
                    timeout=_B2_STAGING_POLL_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    futures.pop(future)
                    future.result()
    except BaseException as exc:
        _write_staging_error(staging_root_path, split, exc)
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
        download_workers: int | None = None,
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
        self.download_workers = resolve_b2_download_workers(download_workers)
        source_key = str(abs(hash((self.source_prefix, os.getpid(), id(self)))))
        self.staging_source_root = self.staging_root / source_key
        self._client = client
        self._remote_reader = B2BlockReader(
            bucket=bucket,
            endpoint_url=endpoint_url,
            key_id=key_id,
            app_key=app_key,
            source_prefix=source_prefix,
            split=split,
            download_workers=self.download_workers,
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
                download_workers=self.download_workers,
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
                download_workers=self.download_workers,
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
                download_workers=self.download_workers,
                client=self._client,
            )
        return self._remote_reader.load_block_metadata_many(cache_keys)

    def prepare_blocks(self, blocks: tuple[dict[str, Any], ...]) -> None:
        cache_keys = tuple(str(block["cache_key"]) for block in blocks)
        remove_tree(self.staging_source_root)
        if self._client is not None:
            if self._remote_reader is None:
                self._remote_reader = B2BlockReader(
                    bucket=self.bucket,
                    endpoint_url=self.endpoint_url,
                    key_id=self.key_id,
                    app_key=self.app_key,
                    source_prefix=self.source_prefix,
                    split=self.split,
                    download_workers=self.download_workers,
                    client=self._client,
                )
            for cache_key in cache_keys:
                self._remote_reader.stage_block(
                    cache_key,
                    staging_source_root=self.staging_source_root,
                    split=self.split,
                )
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
                "download_workers": self.download_workers,
            },
            daemon=True,
        )
        self._process.start()
        self._process_owner_pid = os.getpid()

    def load_block(self, cache_key: str) -> MappedBlockPayload:
        while not v15_block_is_cached(self.staging_source_root, self.split, cache_key):
            _raise_if_staging_error(self.staging_source_root, self.split)
            process = self._process if self._process_owner_pid == os.getpid() else None
            if process is not None and not process.is_alive() and process.exitcode not in (None, 0):
                raise RuntimeError(f"B2 staging downloader exited with code {process.exitcode}")
            time.sleep(_B2_STAGING_POLL_SECONDS)
        return load_v15_block(self.staging_source_root, self.split, cache_key)

    def release_block(self, cache_key: str) -> None:
        clear_v15_block(self.staging_source_root, self.split, cache_key)


def resolve_b2_cache_repository(
    *, prefix: str | os.PathLike[str] | None = None, download_workers: int | None = None
) -> B2CacheRepository:
    return B2CacheRepository.from_env(prefix=prefix, download_workers=download_workers)


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
    "resolve_b2_download_workers",
    "resolve_b2_cache_repository",
]
