from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Any, TypeVar

from datasets import config as datasets_config
from datasets import load_dataset, load_dataset_builder
from datasets.download.download_config import DownloadConfig
from datasets.iterable_dataset import IterableDataset
from datasets.packaged_modules.parquet.parquet import Key, ds as parquet_ds, logger as parquet_logger, pa, pq
from datasets.utils.file_utils import (
    CONNECTION_ERRORS_TO_RETRY,
    RATE_LIMIT_CODE,
    xopen,
)
from huggingface_hub.errors import HfHubHTTPError

from .constants import (
    BLOCK_SIZE,
    CACHE_LAYOUT_VERSION,
    DATA_COLUMNS,
    HF_DATASETS_VERSION,
    REMOTE_DOWNLOAD_MAX_RETRIES,
    REMOTE_RETRY_BASE_DELAY_SEC,
    REMOTE_RETRY_JITTER_RATIO,
    REMOTE_RETRY_MAX_ATTEMPTS,
    REMOTE_RETRY_MAX_DELAY_SEC,
    REMOTE_STREAMING_OPEN_MAX_RETRIES,
    REMOTE_STREAMING_OPEN_RETRY_INTERVAL_SEC,
    REMOTE_STREAMING_READ_MAX_RETRIES,
    REMOTE_STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL_SEC,
    REMOTE_STREAMING_READ_RETRY_INTERVAL_SEC,
    REMOTE_STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL_SEC,
    StartupCallback,
)
from .store import freeze_row, read_json, write_json_atomic


T = TypeVar("T")
_REMOTE_RETRY_STAGE = "remote retry"


def emit_startup_event(startup_callback: StartupCallback | None, **event: Any) -> None:
    if startup_callback is not None:
        startup_callback(event)


def run_startup_stage(
    startup_callback: StartupCallback | None,
    *,
    stage: str,
    split: str,
    operation,
    **fields: Any,
):
    """Wrap an operation in startup lifecycle events."""
    emit_startup_event(startup_callback, stage=stage, split=split, status="start", **fields)
    started_at = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        emit_startup_event(
            startup_callback,
            stage=stage,
            split=split,
            status="error",
            elapsed_sec=time.perf_counter() - started_at,
            error=str(exc),
            **fields,
        )
        raise

    emit_startup_event(
        startup_callback,
        stage=stage,
        split=split,
        status="done",
        elapsed_sec=time.perf_counter() - started_at,
        **fields,
    )
    return result


def resolve_layout_root(cache_root: Path) -> Path:
    path = cache_root / f"layout-v{CACHE_LAYOUT_VERSION}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_source_root(cache_root: Path, source_signature: str) -> Path:
    path = resolve_layout_root(cache_root) / source_signature
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_catalog_root(source_root: Path) -> Path:
    path = source_root / "catalogs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_catalog_path(source_root: Path, split: str) -> Path:
    return resolve_catalog_root(source_root) / f"{split}.json"


def resolve_source_metadata_path(source_root: Path) -> Path:
    return source_root / "source.json"


@dataclass(frozen=True, slots=True)
class BlockDescriptor:
    index: int
    shard_index: int
    cache_key: str
    source_file: str
    row_groups: tuple[int, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "shard_index": self.shard_index,
            "cache_key": self.cache_key,
            "source_file": self.source_file,
            "row_groups": list(self.row_groups),
        }


_source_descriptor_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
_stream_template_cache: dict[tuple[str, str | None, str], IterableDataset] = {}
_PATCHED_GENERATE_TABLES_SENTINEL = "__cr_train_patched_generate_tables__"


@dataclass(frozen=True, slots=True)
class RemoteRetryPolicy:
    outer_max_attempts: int = REMOTE_RETRY_MAX_ATTEMPTS
    outer_base_delay_sec: float = REMOTE_RETRY_BASE_DELAY_SEC
    outer_max_delay_sec: float = REMOTE_RETRY_MAX_DELAY_SEC
    outer_jitter_ratio: float = REMOTE_RETRY_JITTER_RATIO
    download_max_retries: int = REMOTE_DOWNLOAD_MAX_RETRIES
    resume_download: bool = True
    streaming_open_max_retries: int = REMOTE_STREAMING_OPEN_MAX_RETRIES
    streaming_open_retry_interval_sec: float = REMOTE_STREAMING_OPEN_RETRY_INTERVAL_SEC
    streaming_read_max_retries: int = REMOTE_STREAMING_READ_MAX_RETRIES
    streaming_read_retry_interval_sec: float = REMOTE_STREAMING_READ_RETRY_INTERVAL_SEC
    streaming_read_server_unavailable_retry_interval_sec: float = (
        REMOTE_STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL_SEC
    )
    streaming_read_rate_limit_retry_interval_sec: float = (
        REMOTE_STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL_SEC
    )


_DEFAULT_REMOTE_RETRY_POLICY = RemoteRetryPolicy()


def _installed_datasets_version() -> str:
    return importlib.metadata.version("datasets")


def ensure_supported_datasets_version() -> None:
    installed = _installed_datasets_version()
    if installed != HF_DATASETS_VERSION:
        raise RuntimeError(
            "cr-train requires datasets=="
            f"{HF_DATASETS_VERSION} for the HF row-group streaming adapter, found {installed}"
        )


def _resolve_retry_policy(retry_policy: RemoteRetryPolicy | None) -> RemoteRetryPolicy:
    return retry_policy if retry_policy is not None else _DEFAULT_REMOTE_RETRY_POLICY


def _single_attempt_retry_policy(retry_policy: RemoteRetryPolicy) -> RemoteRetryPolicy:
    return replace(retry_policy, outer_max_attempts=1)


def _retryable_http_status(exc: HfHubHTTPError) -> int | None:
    if exc.response is None:
        return None
    status_code = exc.response.status_code
    if status_code == RATE_LIMIT_CODE or 500 <= status_code < 600:
        return status_code
    return None


def _is_retryable_remote_error(exc: Exception) -> bool:
    if isinstance(exc, CONNECTION_ERRORS_TO_RETRY):
        return True
    if isinstance(exc, ConnectionError) and str(exc) == "Server Disconnected":
        return True
    if isinstance(exc, HfHubHTTPError) and _retryable_http_status(exc) is not None:
        return True
    return False


def _compute_retry_delay_sec(retry_policy: RemoteRetryPolicy, attempt: int) -> float:
    base_delay = min(
        retry_policy.outer_max_delay_sec,
        retry_policy.outer_base_delay_sec * (2 ** max(attempt - 1, 0)),
    )
    jitter = random.uniform(0.0, base_delay * retry_policy.outer_jitter_ratio)
    return min(retry_policy.outer_max_delay_sec, base_delay + jitter)


@contextmanager
def _override_streaming_retry_config(retry_policy: RemoteRetryPolicy):
    original_values = {
        "STREAMING_OPEN_MAX_RETRIES": datasets_config.STREAMING_OPEN_MAX_RETRIES,
        "STREAMING_OPEN_RETRY_INTERVAL": datasets_config.STREAMING_OPEN_RETRY_INTERVAL,
        "STREAMING_READ_MAX_RETRIES": datasets_config.STREAMING_READ_MAX_RETRIES,
        "STREAMING_READ_RETRY_INTERVAL": datasets_config.STREAMING_READ_RETRY_INTERVAL,
        "STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL": datasets_config.STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL,
        "STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL": datasets_config.STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL,
    }
    datasets_config.STREAMING_OPEN_MAX_RETRIES = retry_policy.streaming_open_max_retries
    datasets_config.STREAMING_OPEN_RETRY_INTERVAL = retry_policy.streaming_open_retry_interval_sec
    datasets_config.STREAMING_READ_MAX_RETRIES = retry_policy.streaming_read_max_retries
    datasets_config.STREAMING_READ_RETRY_INTERVAL = retry_policy.streaming_read_retry_interval_sec
    datasets_config.STREAMING_READ_SERVER_UNAVAILABLE_RETRY_INTERVAL = (
        retry_policy.streaming_read_server_unavailable_retry_interval_sec
    )
    datasets_config.STREAMING_READ_RATE_LIMIT_RETRY_INTERVAL = (
        retry_policy.streaming_read_rate_limit_retry_interval_sec
    )
    try:
        yield
    finally:
        for name, value in original_values.items():
            setattr(datasets_config, name, value)


def _build_download_config(
    retry_policy: RemoteRetryPolicy,
    *,
    token: str | bool | None = None,
    storage_options: dict[str, Any] | None = None,
) -> DownloadConfig:
    return DownloadConfig(
        token=token,
        storage_options=dict(storage_options or {}),
        max_retries=retry_policy.download_max_retries,
        resume_download=retry_policy.resume_download,
    )


def _format_retry_context(
    *,
    split: str | None,
    context_fields: dict[str, Any],
) -> str:
    parts: list[str] = []
    if split is not None:
        parts.append(f"split={split}")
    for key in ("dataset_name", "revision", "cache_key", "shard_index", "source_file", "row_groups"):
        value = context_fields.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _emit_remote_retry(
    startup_callback: StartupCallback | None,
    *,
    operation: str,
    attempt: int,
    max_attempts: int,
    delay_sec: float,
    split: str | None,
    error: Exception,
    context_fields: dict[str, Any],
) -> None:
    event: dict[str, Any] = {
        "stage": _REMOTE_RETRY_STAGE,
        "status": "retry",
        "operation": operation,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "delay_sec": delay_sec,
        "error_type": type(error).__name__,
        "error": str(error),
        **context_fields,
    }
    if split is not None:
        event["split"] = split
    emit_startup_event(startup_callback, **event)


def _with_remote_retries(
    operation: Callable[[], T],
    *,
    operation_name: str,
    retry_policy: RemoteRetryPolicy | None = None,
    startup_callback: StartupCallback | None = None,
    split: str | None = None,
    on_retry_reset: Callable[[], None] | None = None,
    context_fields: dict[str, Any] | None = None,
) -> T:
    resolved_policy = _resolve_retry_policy(retry_policy)
    resolved_context = dict(context_fields or {})
    last_retryable_error: Exception | None = None

    for attempt in range(1, resolved_policy.outer_max_attempts + 1):
        try:
            with _override_streaming_retry_config(resolved_policy):
                return operation()
        except Exception as exc:
            if not _is_retryable_remote_error(exc):
                raise
            last_retryable_error = exc
            if attempt >= resolved_policy.outer_max_attempts:
                break
            delay_sec = _compute_retry_delay_sec(resolved_policy, attempt)
            _emit_remote_retry(
                startup_callback,
                operation=operation_name,
                attempt=attempt,
                max_attempts=resolved_policy.outer_max_attempts,
                delay_sec=delay_sec,
                split=split,
                error=exc,
                context_fields=resolved_context,
            )
            if on_retry_reset is not None:
                on_retry_reset()
            time.sleep(delay_sec)

    if last_retryable_error is None:
        raise RuntimeError(f"{operation_name} failed without a retryable error")
    if resolved_policy.outer_max_attempts <= 1:
        raise last_retryable_error

    context = _format_retry_context(split=split, context_fields=resolved_context)
    detail = f"{type(last_retryable_error).__name__}: {last_retryable_error}"
    if context:
        raise RuntimeError(
            f"{operation_name} failed after {resolved_policy.outer_max_attempts} attempts ({context}): {detail}"
        ) from last_retryable_error
    raise RuntimeError(
        f"{operation_name} failed after {resolved_policy.outer_max_attempts} attempts: {detail}"
    ) from last_retryable_error


def load_source_descriptor(
    dataset_name: str,
    revision: str | None,
    *,
    startup_callback: StartupCallback | None = None,
    retry_policy: RemoteRetryPolicy | None = None,
) -> dict[str, Any]:
    ensure_supported_datasets_version()
    cache_key = (dataset_name, revision)
    cached = _source_descriptor_cache.get(cache_key)
    if cached is not None:
        return cached

    resolved_policy = _resolve_retry_policy(retry_policy)

    def _load_once() -> dict[str, Any]:
        builder = load_dataset_builder(
            dataset_name,
            revision=revision,
            download_config=_build_download_config(resolved_policy),
        )
        split_sizes = {
            str(split_name): int(split_info.num_examples)
            for split_name, split_info in builder.info.splits.items()
        }
        signature_payload = {
            "cache_layout_version": CACHE_LAYOUT_VERSION,
            "dataset_name": dataset_name,
            "revision": revision,
            "split_sizes": split_sizes,
        }
        source_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        return {
            "dataset_name": dataset_name,
            "revision": revision,
            "source_signature": source_signature,
            "split_sizes": split_sizes,
        }

    descriptor = _with_remote_retries(
        _load_once,
        operation_name="load source descriptor",
        retry_policy=resolved_policy,
        startup_callback=startup_callback,
        split=None,
        on_retry_reset=lambda: _source_descriptor_cache.pop(cache_key, None),
        context_fields={
            "dataset_name": dataset_name,
            "revision": revision,
        },
    )
    _source_descriptor_cache[cache_key] = descriptor
    return descriptor


def _find_cached_source(
    cache_root: Path,
    dataset_name: str,
    revision: str | None,
) -> tuple[Path, dict[str, Any]] | None:
    layout_root = resolve_layout_root(cache_root)
    if not layout_root.is_dir():
        return None
    for entry in layout_root.iterdir():
        if not entry.is_dir():
            continue
        metadata_path = resolve_source_metadata_path(entry)
        if not metadata_path.exists():
            continue
        try:
            metadata = read_json(metadata_path)
        except Exception:
            continue
        if metadata.get("dataset_name") == dataset_name and metadata.get("revision") == revision:
            return entry, metadata
    return None


def ensure_source_root(
    *,
    dataset_name: str,
    revision: str | None,
    cache_root: Path,
    startup_callback: StartupCallback | None = None,
    retry_policy: RemoteRetryPolicy | None = None,
) -> tuple[Path, dict[str, Any]]:
    cached = _find_cached_source(cache_root, dataset_name, revision)
    try:
        descriptor = load_source_descriptor(
            dataset_name,
            revision,
            startup_callback=startup_callback,
            retry_policy=retry_policy,
        )
    except Exception:
        if cached is None:
            raise
        source_root, cached_descriptor = cached
        _source_descriptor_cache[(dataset_name, revision)] = cached_descriptor
        return source_root, cached_descriptor

    source_root = resolve_source_root(cache_root, str(descriptor["source_signature"]))
    metadata_path = resolve_source_metadata_path(source_root)
    payload = {
        "cache_layout_version": CACHE_LAYOUT_VERSION,
        **descriptor,
    }
    if not metadata_path.exists():
        write_json_atomic(metadata_path, payload)
    else:
        try:
            existing = read_json(metadata_path)
        except Exception:
            existing = None
        if existing != payload:
            write_json_atomic(metadata_path, payload)
    return source_root, descriptor


def _clone_iterable_dataset(dataset: IterableDataset, kwargs: dict[str, Any]) -> IterableDataset:
    ex_iterable = dataset._ex_iterable
    cloned = type(ex_iterable)(
        ex_iterable.generate_tables_fn,
        kwargs,
        ex_iterable.generate_more_kwargs_fn,
    )
    info = dataset._info.copy() if hasattr(dataset._info, "copy") else dataset._info
    return IterableDataset(
        ex_iterable=cloned,
        info=info,
        split=dataset._split,
        formatting=getattr(dataset, "_formatting", None),
        distributed=getattr(dataset, "_distributed", None),
        token_per_repo_id=getattr(dataset, "_token_per_repo_id", None),
    )


def _patch_parquet_generate_tables_fn(
    dataset: IterableDataset,
    retry_policy: RemoteRetryPolicy | None = None,
) -> None:
    ex_iterable = getattr(dataset, "_ex_iterable", None)
    generate_tables_fn = getattr(ex_iterable, "generate_tables_fn", None)
    builder = getattr(generate_tables_fn, "__self__", None)
    generate_tables_func = getattr(generate_tables_fn, "__func__", generate_tables_fn)
    if generate_tables_fn is None or builder is None:
        raise RuntimeError("unexpected HF parquet iterable layout: missing generate_tables_fn")
    if getattr(generate_tables_func, _PATCHED_GENERATE_TABLES_SENTINEL, False):
        return
    resolved_policy = _resolve_retry_policy(retry_policy)

    def generate_tables_wrapper(self, files, row_groups_list):
        if self.config.features is not None and self.config.columns is not None:
            if sorted(field.name for field in self.info.features.arrow_schema) != sorted(self.config.columns):
                raise ValueError(
                    f"Tried to load parquet data with columns '{self.config.columns}' with mismatching features '{self.info.features}'"
                )
        filter_expr = (
            pq.filters_to_expression(self.config.filters)
            if isinstance(self.config.filters, list)
            else self.config.filters
        )
        parquet_file_format = parquet_ds.ParquetFileFormat(
            default_fragment_scan_options=self.config.fragment_scan_options
        )
        download_config = _build_download_config(
            resolved_policy,
            token=getattr(self, "token", None),
            storage_options=dict(getattr(self, "storage_options", {}) or {}),
        )
        for file_idx, (file, row_groups) in enumerate(zip(files, row_groups_list)):
            try:
                with xopen(file, "rb", download_config=download_config) as f:
                    parquet_fragment = parquet_file_format.make_fragment(f)
                    if row_groups is not None:
                        parquet_fragment = parquet_fragment.subset(row_group_ids=row_groups)
                    if parquet_fragment.row_groups:
                        batch_size = self.config.batch_size or parquet_fragment.row_groups[0].num_rows
                        for batch_idx, record_batch in enumerate(
                            parquet_fragment.to_batches(
                                batch_size=batch_size,
                                columns=self.config.columns,
                                filter=filter_expr,
                                batch_readahead=0,
                                fragment_readahead=0,
                            )
                        ):
                            pa_table = pa.Table.from_batches([record_batch])
                            yield Key(file_idx, batch_idx), self._cast_table(pa_table)
            except (pa.ArrowInvalid, ValueError) as e:
                if self.config.on_bad_files == "error":
                    parquet_logger.error(f"Failed to read file '{file}' with error {type(e).__name__}: {e}")
                    raise
                if self.config.on_bad_files == "warn":
                    parquet_logger.warning(f"Skipping bad file '{file}'. {type(e).__name__}: {e}`")
                else:
                    parquet_logger.debug(f"Skipping bad file '{file}'. {type(e).__name__}: {e}`")

    generate_tables_wrapper.__name__ = getattr(generate_tables_func, "__name__", "generate_tables_wrapper")
    generate_tables_wrapper.__qualname__ = getattr(
        generate_tables_func,
        "__qualname__",
        generate_tables_wrapper.__qualname__,
    )
    setattr(generate_tables_wrapper, _PATCHED_GENERATE_TABLES_SENTINEL, True)
    ex_iterable.generate_tables_fn = MethodType(generate_tables_wrapper, builder)


def _prepare_row_group_stream(
    dataset: IterableDataset,
    *,
    retry_policy: RemoteRetryPolicy | None = None,
) -> IterableDataset:
    _patch_parquet_generate_tables_fn(dataset, retry_policy=retry_policy)
    ex_iterable = getattr(dataset, "_ex_iterable", None)
    kwargs = dict(getattr(ex_iterable, "kwargs", {}))
    if not kwargs or "files" not in kwargs or "row_groups_list" not in kwargs:
        raise RuntimeError("unexpected HF iterable layout: missing parquet kwargs")

    row_groups_list = kwargs["row_groups_list"]
    if row_groups_list is None:
        return dataset.reshard()

    if not isinstance(row_groups_list, list) or not all(item is None for item in row_groups_list):
        return dataset.reshard()

    base_num_shards = int(dataset.num_shards)
    cloned_kwargs = dict(kwargs)
    cloned_kwargs["row_groups_list"] = None
    row_group_dataset = _clone_iterable_dataset(dataset, cloned_kwargs)
    resharded = row_group_dataset.reshard()
    if int(resharded.num_shards) <= base_num_shards:
        raise RuntimeError("HF parquet reshard did not expose row-group shards for this dataset")
    return resharded


def _stream_cache_key(dataset_name: str, revision: str | None, split: str) -> tuple[str, str | None, str]:
    return dataset_name, revision, split


def _clear_stream_template_cache(dataset_name: str, revision: str | None, split: str) -> None:
    _stream_template_cache.pop(_stream_cache_key(dataset_name, revision, split), None)


def _load_row_group_stream_once(
    *,
    dataset_name: str,
    revision: str | None,
    split: str,
    retry_policy: RemoteRetryPolicy,
) -> IterableDataset:
    cache_key = _stream_cache_key(dataset_name, revision, split)
    cached = _stream_template_cache.get(cache_key)
    if cached is not None:
        return cached

    stream = load_dataset(
        dataset_name,
        revision=revision,
        split=split,
        streaming=True,
        columns=DATA_COLUMNS,
        download_config=_build_download_config(retry_policy),
    )
    row_group_stream = _prepare_row_group_stream(stream, retry_policy=retry_policy)
    _stream_template_cache[cache_key] = row_group_stream
    return row_group_stream


def load_row_group_stream(
    *,
    dataset_name: str,
    revision: str | None,
    split: str,
    startup_callback: StartupCallback | None = None,
    retry_policy: RemoteRetryPolicy | None = None,
) -> IterableDataset:
    ensure_supported_datasets_version()
    resolved_policy = _resolve_retry_policy(retry_policy)
    return _with_remote_retries(
        lambda: _load_row_group_stream_once(
            dataset_name=dataset_name,
            revision=revision,
            split=split,
            retry_policy=resolved_policy,
        ),
        operation_name="load row-group stream",
        retry_policy=resolved_policy,
        startup_callback=startup_callback,
        split=split,
        on_retry_reset=lambda: _clear_stream_template_cache(dataset_name, revision, split),
        context_fields={
            "dataset_name": dataset_name,
            "revision": revision,
        },
    )


def _build_block_descriptor(
    *,
    dataset_name: str,
    revision: str | None,
    split: str,
    index: int,
    shard_index: int,
    source_file: str,
    row_groups: tuple[int, ...],
) -> BlockDescriptor:
    payload = {
        "dataset_name": dataset_name,
        "revision": revision,
        "split": split,
        "source_file": source_file,
        "row_groups": list(row_groups),
    }
    cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:32]
    return BlockDescriptor(
        index=index,
        shard_index=shard_index,
        cache_key=cache_key,
        source_file=source_file,
        row_groups=row_groups,
    )


def build_catalog(
    *,
    dataset_name: str,
    revision: str | None,
    split: str,
    total_rows: int,
    startup_callback: StartupCallback | None = None,
    retry_policy: RemoteRetryPolicy | None = None,
) -> dict[str, Any]:
    stream = load_row_group_stream(
        dataset_name=dataset_name,
        revision=revision,
        split=split,
        startup_callback=startup_callback,
        retry_policy=retry_policy,
    )
    ex_iterable = stream._ex_iterable
    kwargs = dict(getattr(ex_iterable, "kwargs", {}))
    files = list(kwargs.get("files", []))
    row_groups_list = kwargs.get("row_groups_list")
    if row_groups_list is None or len(files) != len(row_groups_list):
        raise RuntimeError("unexpected row-group iterable layout after reshard")

    blocks = []
    for index, (source_file, row_groups) in enumerate(zip(files, row_groups_list, strict=True)):
        if row_groups is None:
            raise RuntimeError("unexpected unsplit parquet shard after row-group reshard")
        block = _build_block_descriptor(
            dataset_name=dataset_name,
            revision=revision,
            split=split,
            index=index,
            shard_index=index,
            source_file=str(source_file),
            row_groups=tuple(int(value) for value in row_groups),
        )
        blocks.append(block.to_payload())

    return {
        "cache_layout_version": CACHE_LAYOUT_VERSION,
        "split": split,
        "total_rows": total_rows,
        "total_blocks": len(blocks),
        "blocks": blocks,
    }


def _write_and_reload_catalog(*, catalog_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    write_json_atomic(catalog_path, payload)
    return read_json(catalog_path)


def ensure_split_catalog(
    *,
    source_root: Path,
    descriptor: dict[str, Any],
    split: str,
    startup_callback: StartupCallback | None,
    retry_policy: RemoteRetryPolicy | None = None,
) -> dict[str, Any]:
    catalog_path = resolve_catalog_path(source_root, split)
    split_sizes = descriptor["split_sizes"]
    if split not in split_sizes:
        raise KeyError(f"split {split!r} does not exist in source descriptor")
    cached_catalog = read_json(catalog_path) if catalog_path.exists() else None

    try:
        payload = run_startup_stage(
            startup_callback,
            stage="ensure catalog",
            split=split,
            operation=lambda: build_catalog(
                dataset_name=str(descriptor["dataset_name"]),
                revision=descriptor.get("revision"),
                split=split,
                total_rows=int(split_sizes[split]),
                startup_callback=startup_callback,
                retry_policy=retry_policy,
            ),
            total_rows=int(split_sizes[split]),
        )
    except Exception:
        if cached_catalog is not None:
            return cached_catalog
        raise

    if cached_catalog != payload:
        return _write_and_reload_catalog(catalog_path=catalog_path, payload=payload)
    return cached_catalog if cached_catalog is not None else payload


def _block_error_context(block: dict[str, Any], *, shard_index: int) -> str:
    return (
        f"cache_key={block['cache_key']} "
        f"shard_index={shard_index} "
        f"source_file={block['source_file']} "
        f"row_groups={list(block['row_groups'])}"
    )


def _load_block_rows_once(
    *,
    dataset_name: str,
    revision: str | None,
    split: str,
    block: dict[str, Any],
    progress_callback: Callable[[int, int], None] | None,
    retry_policy: RemoteRetryPolicy,
) -> list[dict[str, Any]]:
    template = load_row_group_stream(
        dataset_name=dataset_name,
        revision=revision,
        split=split,
        retry_policy=_single_attempt_retry_policy(retry_policy),
    )
    shard_index = int(block["shard_index"])
    block_dataset = template.shard(
        num_shards=int(template.num_shards),
        index=shard_index,
        contiguous=True,
    )
    raw_rows = list(block_dataset.take(BLOCK_SIZE + 1))
    row_count = len(raw_rows)
    block_context = _block_error_context(block, shard_index=shard_index)
    if row_count <= 0:
        raise RuntimeError(f"empty row-group shard: {block_context}")
    if row_count > BLOCK_SIZE:
        raise RuntimeError(
            f"row-group shard exceeded BLOCK_SIZE={BLOCK_SIZE}: got {row_count}; {block_context}"
        )

    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(raw_rows, start=1):
        frozen = freeze_row(dict(row))
        rows.append(frozen)
        if progress_callback is not None:
            downloaded_bytes = sum(
                len(value)
                for value in frozen.values()
                if isinstance(value, (bytes, bytearray, memoryview))
            )
            progress_callback(row_index, downloaded_bytes)
    return rows


def load_block_rows(
    *,
    dataset_name: str,
    revision: str | None,
    split: str,
    block: dict[str, Any],
    progress_callback: Callable[[int, int], None] | None = None,
    startup_callback: StartupCallback | None = None,
    retry_policy: RemoteRetryPolicy | None = None,
) -> list[dict[str, Any]]:
    resolved_policy = _resolve_retry_policy(retry_policy)
    context_fields = {
        "dataset_name": dataset_name,
        "revision": revision,
        "cache_key": str(block["cache_key"]),
        "shard_index": int(block["shard_index"]),
        "source_file": str(block["source_file"]),
        "row_groups": list(block["row_groups"]),
    }
    return _with_remote_retries(
        lambda: _load_block_rows_once(
            dataset_name=dataset_name,
            revision=revision,
            split=split,
            block=block,
            progress_callback=progress_callback,
            retry_policy=resolved_policy,
        ),
        operation_name="load block rows",
        retry_policy=resolved_policy,
        startup_callback=startup_callback,
        split=split,
        on_retry_reset=lambda: _clear_stream_template_cache(dataset_name, revision, split),
        context_fields=context_fields,
    )


__all__ = [
    "BlockDescriptor",
    "emit_startup_event",
    "ensure_source_root",
    "ensure_split_catalog",
    "load_block_rows",
    "load_row_group_stream",
    "resolve_catalog_path",
    "run_startup_stage",
]
