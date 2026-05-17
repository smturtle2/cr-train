from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue as queue_mod
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import zstandard as zstd
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import get_session

from .constants import CRPACK_BLOCK_SIZE, CRPACK_LAYOUT_VERSION
from .runtime import emit_startup_event
from .store import MappedBlockPayload, file_lock, read_json, remove_tree, write_json_atomic
from .v15 import (
    CRPACK_EXTENSION,
    clear_v15_block,
    load_v15_block,
    read_v15_header_from_file,
    unpack_v15_block_bytes,
    v15_block_is_cached,
    v15_block_path,
    write_v15_block_bytes,
)


HF_V2_DATASET_ID = "Hermanni/sen12mscr-v2"
HF_V2_REVISION = "main"
HF_V2_LAYOUT = "cr-hf-scene-v1"
HF_V2_DOWNLOAD_WORKERS = 16
HF_V2_STAGING_MAX_BLOCKS = 80
_HF_V2_READ_ATTEMPTS = 4
_HF_V2_RETRY_BASE_DELAY_SECONDS = 0.25
_HF_V2_STAGING_POLL_SECONDS = 0.05


class BlockReader(Protocol):
    def load_block(self, cache_key: str) -> MappedBlockPayload:
        ...


@dataclass(frozen=True, slots=True)
class HFV2Block:
    cache_key: str
    path: str
    row_count: int
    index: int
    bytes: int | None = None


def resolve_dataset_root(dataset_dir: str | os.PathLike[str] | None) -> Path:
    return (
        Path(dataset_dir)
        if dataset_dir is not None
        else Path.home() / ".cache" / "cr-train" / "sen12mscr-v2"
    )


def resolve_streaming_stage_root() -> Path:
    return Path.home() / ".cache" / "cr-train" / "streaming-stage"


def hf_v2_download_worker_count() -> int:
    return HF_V2_DOWNLOAD_WORKERS


def resolve_hf_v2_download_workers(download_workers: int | None = None) -> int:
    if download_workers is None:
        return HF_V2_DOWNLOAD_WORKERS
    resolved = int(download_workers)
    if resolved <= 0:
        raise ValueError("download_workers must be greater than zero")
    return resolved


def _hf_url(path: str) -> str:
    return hf_hub_url(
        HF_V2_DATASET_ID,
        path.lstrip("/"),
        repo_type="dataset",
        revision=HF_V2_REVISION,
    )


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(_HF_V2_RETRY_BASE_DELAY_SECONDS * (2 ** attempt))


def _read_remote_bytes(path: str) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(_HF_V2_READ_ATTEMPTS):
        try:
            response = get_session().get(_hf_url(path), follow_redirects=True)
            response.raise_for_status()
            return bytes(response.content)
        except Exception as exc:
            last_exc = exc
            if attempt < _HF_V2_READ_ATTEMPTS - 1:
                _sleep_before_retry(attempt)
                continue
            break
    raise RuntimeError(f"failed to read HF dataset object: {path}") from last_exc


def _download_remote_file(path: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    remove_tree(tmp_path)
    last_exc: Exception | None = None
    for attempt in range(_HF_V2_READ_ATTEMPTS):
        try:
            downloaded = 0
            with get_session().stream("GET", _hf_url(path), follow_redirects=True) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
            tmp_path.replace(target)
            return downloaded
        except Exception as exc:
            last_exc = exc
            remove_tree(tmp_path)
            if attempt < _HF_V2_READ_ATTEMPTS - 1:
                _sleep_before_retry(attempt)
                continue
            break
    raise RuntimeError(f"failed to download HF dataset object: {path}") from last_exc


def _read_remote_json(path: str) -> dict[str, Any]:
    try:
        return json.loads(_read_remote_bytes(path).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"HF dataset object is not valid JSON: {path}") from exc


def _download_json_if_missing(dataset_root: Path, path: str) -> dict[str, Any]:
    local_path = dataset_root / path
    if local_path.exists():
        return read_json(local_path)
    payload = _read_remote_json(path)
    write_json_atomic(local_path, payload)
    return payload


def load_hf_v2_manifest(*, dataset_root: Path | None = None, streaming: bool = True) -> dict[str, Any]:
    manifest = (
        _read_remote_json("manifest.json")
        if streaming
        else _download_json_if_missing(_require_dataset_root(dataset_root), "manifest.json")
    )
    if manifest.get("layout") != HF_V2_LAYOUT:
        raise RuntimeError(f"unsupported HF v2 dataset layout: {manifest.get('layout')!r}")
    if int(manifest.get("block_format_version", 0)) != CRPACK_LAYOUT_VERSION:
        raise RuntimeError(
            f"unsupported HF v2 block format version: {manifest.get('block_format_version')!r}"
        )
    return manifest


def load_hf_v2_split_catalog(
    *,
    split: str,
    dataset_root: Path | None = None,
    streaming: bool = True,
) -> dict[str, Any]:
    path = f"catalogs/{split}.json"
    catalog = (
        _read_remote_json(path)
        if streaming
        else _download_json_if_missing(_require_dataset_root(dataset_root), path)
    )
    return normalize_hf_v2_catalog(catalog, split=split)


def normalize_hf_v2_catalog(catalog: dict[str, Any], *, split: str) -> dict[str, Any]:
    raw_blocks = list(catalog.get("blocks", []))
    blocks: list[dict[str, Any]] = []
    block_row_counts: list[int] = []
    for fallback_index, block in enumerate(raw_blocks):
        cache_key = str(block.get("cache_key") or block.get("block_id"))
        if not cache_key:
            raise RuntimeError(f"HF v2 catalog block is missing block_id/cache_key: split={split}")
        row_count = int(block["row_count"])
        index = int(block.get("global_index", block.get("index", fallback_index)))
        normalized = dict(block)
        normalized["cache_key"] = cache_key
        normalized["block_id"] = cache_key
        normalized["index"] = index
        normalized["row_count"] = row_count
        if "path" not in normalized:
            raise RuntimeError(f"HF v2 catalog block is missing path: cache_key={cache_key}")
        blocks.append(normalized)
        block_row_counts.append(row_count)
    return {
        "cache_layout_version": CRPACK_LAYOUT_VERSION,
        "cache_block_size": int(catalog.get("cache_block_size", CRPACK_BLOCK_SIZE)),
        "split": split,
        "total_rows": int(catalog.get("row_count", sum(block_row_counts))),
        "total_blocks": int(catalog.get("block_count", len(blocks))),
        "block_row_counts": block_row_counts,
        "blocks": blocks,
    }


def _require_dataset_root(dataset_root: Path | None) -> Path:
    if dataset_root is None:
        raise ValueError("dataset_root must be provided")
    return Path(dataset_root)


def _block_path_by_key(blocks: tuple[dict[str, Any], ...]) -> dict[str, str]:
    return {str(block["cache_key"]): str(block["path"]) for block in blocks}


def _row_count_by_key(blocks: tuple[dict[str, Any], ...]) -> dict[str, int]:
    return {str(block["cache_key"]): int(block["row_count"]) for block in blocks}


def _local_block_path(dataset_root: Path, relative_path: str) -> Path:
    return dataset_root / relative_path


def _block_header_matches(path: Path, *, expected_row_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        header = read_v15_header_from_file(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        int(header.get("version", 0)) == CRPACK_LAYOUT_VERSION
        and int(header.get("row_count", 0)) == int(expected_row_count)
    )


def _completion_root(dataset_root: Path) -> Path:
    return dataset_root / ".cr-train"


def _completion_path(dataset_root: Path, split: str) -> Path:
    return _completion_root(dataset_root) / f"{split}.complete.json"


def is_hf_v2_full_split_complete(dataset_root: Path, split: str) -> bool:
    return _completion_path(dataset_root, split).is_file()


def mark_hf_v2_full_split_complete(
    *,
    dataset_root: Path,
    split: str,
    catalog: dict[str, Any],
) -> None:
    write_json_atomic(
        _completion_path(dataset_root, split),
        {
            "dataset": HF_V2_DATASET_ID,
            "revision": HF_V2_REVISION,
            "layout": HF_V2_LAYOUT,
            "block_format_version": CRPACK_LAYOUT_VERSION,
            "split": split,
            "row_count": int(catalog["total_rows"]),
            "block_count": int(catalog["total_blocks"]),
        },
    )


def _emit_local_summary(
    startup_callback,
    *,
    split: str,
    status: str,
    requested_rows: int,
    effective_rows: int,
    required_blocks: int,
    planner_mode: str,
    selected_block_count: int,
    selected_missing_blocks: int,
    execution_block_count: int | None = None,
    resolved_blocks: int = 0,
    elapsed_sec: float | None = None,
    timeline: str | None = None,
) -> None:
    event: dict[str, Any] = {
        "stage": "warm local dataset",
        "split": split,
        "status": status,
        "requested_rows": requested_rows,
        "effective_rows": effective_rows,
        "required_blocks": required_blocks,
        "planner_mode": planner_mode,
        "selected_block_count": selected_block_count,
        "ready_selected_blocks": selected_block_count - selected_missing_blocks,
        "selected_missing_blocks": selected_missing_blocks,
        "execution_block_count": selected_block_count
        if execution_block_count is None
        else int(execution_block_count),
        "resolved_blocks": resolved_blocks,
    }
    if elapsed_sec is not None:
        event["elapsed_sec"] = elapsed_sec
    if timeline is not None:
        event["timeline"] = timeline
    emit_startup_event(startup_callback, **event)


def ensure_hf_v2_local_blocks(
    *,
    dataset_root: Path,
    split: str,
    catalog: dict[str, Any],
    selected_blocks: tuple[dict[str, Any], ...],
    requested_rows: int,
    effective_rows: int,
    required_blocks: int,
    planner_mode: str,
    execution_block_count: int | None = None,
    full_split: bool,
    timeline: str | None = None,
    startup_callback=None,
) -> None:
    dataset_root = Path(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    if full_split and is_hf_v2_full_split_complete(dataset_root, split):
        _emit_local_summary(
            startup_callback,
            split=split,
            status="start",
            requested_rows=requested_rows,
            effective_rows=effective_rows,
            required_blocks=required_blocks,
            planner_mode=planner_mode,
            selected_block_count=len(selected_blocks),
            selected_missing_blocks=0,
            execution_block_count=execution_block_count,
        )
        _emit_local_summary(
            startup_callback,
            split=split,
            status="done",
            requested_rows=requested_rows,
            effective_rows=effective_rows,
            required_blocks=required_blocks,
            planner_mode=planner_mode,
            selected_block_count=len(selected_blocks),
            selected_missing_blocks=0,
            execution_block_count=execution_block_count,
            elapsed_sec=0.0,
            timeline=timeline,
        )
        return

    missing = [
        block
        for block in selected_blocks
        if not _block_header_matches(
            _local_block_path(dataset_root, str(block["path"])),
            expected_row_count=int(block["row_count"]),
        )
    ]
    _emit_local_summary(
        startup_callback,
        split=split,
        status="start",
        requested_rows=requested_rows,
        effective_rows=effective_rows,
        required_blocks=required_blocks,
        planner_mode=planner_mode,
        selected_block_count=len(selected_blocks),
        selected_missing_blocks=len(missing),
        execution_block_count=execution_block_count,
    )
    started_at = time.perf_counter()
    resolved_blocks = 0
    for block in missing:
        relative_path = str(block["path"])
        target = _local_block_path(dataset_root, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(target.with_suffix(target.suffix + ".lock")):
            if not _block_header_matches(target, expected_row_count=int(block["row_count"])):
                _download_remote_file(relative_path, target)
            resolved_blocks += 1
    if full_split:
        mark_hf_v2_full_split_complete(dataset_root=dataset_root, split=split, catalog=catalog)
    _emit_local_summary(
        startup_callback,
        split=split,
        status="done",
        requested_rows=requested_rows,
        effective_rows=effective_rows,
        required_blocks=required_blocks,
        planner_mode=planner_mode,
        selected_block_count=len(selected_blocks),
        selected_missing_blocks=len(missing),
        execution_block_count=execution_block_count,
        resolved_blocks=resolved_blocks,
        elapsed_sec=time.perf_counter() - started_at,
        timeline=timeline,
    )


@dataclass(slots=True)
class HFV2LocalBlockReader:
    dataset_root: Path
    block_path_by_key: dict[str, str]
    row_count_by_key: dict[str, int]

    def load_block(self, cache_key: str) -> MappedBlockPayload:
        path = _local_block_path(self.dataset_root, self.block_path_by_key[cache_key])
        try:
            payload, _metadata = unpack_v15_block_bytes(path.read_bytes())
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, zstd.ZstdError) as exc:
            raise RuntimeError(f"local dataset block payload is unreadable for {cache_key}") from exc
        return payload

    def block_is_ready(self, cache_key: str) -> bool:
        return _local_block_path(self.dataset_root, self.block_path_by_key[cache_key]).is_file()

    def load_block_metadata(self, cache_key: str) -> dict[str, Any] | None:
        path = _local_block_path(self.dataset_root, self.block_path_by_key[cache_key])
        if not path.exists():
            return None
        try:
            header = read_v15_header_from_file(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return {"row_count": int(header["row_count"])}

    def load_block_metadata_many(self, cache_keys: tuple[str, ...]) -> dict[str, dict[str, Any] | None]:
        return {cache_key: self.load_block_metadata(cache_key) for cache_key in cache_keys}


def _write_staged_crpack(
    *,
    staging_source_root: Path,
    split: str,
    cache_key: str,
    relative_path: str,
    payload: bytes | None = None,
) -> None:
    if payload is None:
        payload = _read_remote_bytes(relative_path)
    write_v15_block_bytes(
        source_root=staging_source_root,
        split=split,
        cache_key=cache_key,
        blob=payload,
    )


def _count_staged_crpacks(staging_source_root: Path, split: str) -> int:
    block_root = staging_source_root / "block_store" / split / "blocks"
    if not block_root.exists():
        return 0
    return sum(1 for path in block_root.glob(f"*{CRPACK_EXTENSION}") if path.is_file())


@dataclass(frozen=True, slots=True)
class _WorkerStagingPlan:
    worker_queues: tuple[tuple[str, ...], ...]
    worker_quotas: tuple[int, ...]
    worker_by_key: dict[str, int]
    effective_max_staged_blocks: int


def _build_worker_staging_plan(
    cache_keys: tuple[str, ...],
    *,
    max_staged_blocks: int,
    worker_count: int,
) -> _WorkerStagingPlan:
    resolved_worker_count = max(1, int(worker_count))
    effective_max_staged_blocks = max(1, int(max_staged_blocks), resolved_worker_count)
    base_quota, extra_slots = divmod(effective_max_staged_blocks, resolved_worker_count)
    worker_quotas = tuple(
        base_quota + (1 if worker_id < extra_slots else 0)
        for worker_id in range(resolved_worker_count)
    )
    worker_queues = tuple(
        tuple(cache_key for index, cache_key in enumerate(cache_keys) if index % resolved_worker_count == worker_id)
        for worker_id in range(resolved_worker_count)
    )
    worker_by_key: dict[str, int] = {}
    for worker_id, worker_queue in enumerate(worker_queues):
        for cache_key in worker_queue:
            worker_by_key[cache_key] = worker_id
    return _WorkerStagingPlan(
        worker_queues=worker_queues,
        worker_quotas=worker_quotas,
        worker_by_key=worker_by_key,
        effective_max_staged_blocks=effective_max_staged_blocks,
    )


def _download_and_stage_crpack(
    *,
    staging_source_root: Path,
    split: str,
    cache_key: str,
    relative_path: str,
) -> str:
    if v15_block_is_cached(staging_source_root, split, cache_key):
        return cache_key
    _write_staged_crpack(
        staging_source_root=staging_source_root,
        split=split,
        cache_key=cache_key,
        relative_path=relative_path,
    )
    return cache_key


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
        raise RuntimeError(f"HF v2 streaming downloader failed: {message}")


@dataclass(slots=True)
class _WorkerStagingRuntime:
    split: str
    staging_root: Path
    block_paths_by_key: dict[str, str]
    worker_queues: tuple[tuple[str, ...], ...]
    worker_quotas: tuple[int, ...]
    release_queue: Any
    max_staged_blocks: int
    queue_positions: list[int] = field(init=False)
    active_by_worker: list[set[str]] = field(init=False)
    active_owner_by_key: dict[str, int] = field(init=False)
    active_order: list[str] = field(init=False)
    released_cache_keys: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.queue_positions = [0 for _queue in self.worker_queues]
        self.active_by_worker = [set() for _queue in self.worker_queues]
        self.active_owner_by_key = {}
        self.active_order = []
        self.released_cache_keys = set()

    def has_pending_work(self) -> bool:
        return bool(self.active_owner_by_key) or any(
            self.queue_positions[worker_id] < len(self.worker_queues[worker_id])
            for worker_id in range(len(self.worker_queues))
        )

    def fill_worker_slots(self) -> None:
        while True:
            added = False
            for worker_id in range(len(self.worker_queues)):
                if self._fill_one_worker_slot(worker_id):
                    added = True
            if not added:
                return

    def _fill_one_worker_slot(self, worker_id: int) -> bool:
        if len(self.active_by_worker[worker_id]) >= self.worker_quotas[worker_id]:
            return False
        worker_queue = self.worker_queues[worker_id]
        if self.queue_positions[worker_id] >= len(worker_queue):
            return False
        cache_key = worker_queue[self.queue_positions[worker_id]]
        self.queue_positions[worker_id] += 1
        self.active_by_worker[worker_id].add(cache_key)
        self.active_owner_by_key[cache_key] = worker_id
        self.active_order.append(cache_key)
        return True

    def drain_released_blocks(self) -> None:
        while True:
            try:
                cache_key = str(self.release_queue.get_nowait())
            except queue_mod.Empty:
                return
            self.released_cache_keys.add(cache_key)
            clear_v15_block(self.staging_root, self.split, cache_key)
            worker_id = self.active_owner_by_key.pop(cache_key, None)
            if worker_id is None:
                continue
            self.active_by_worker[worker_id].discard(cache_key)

    def submit_downloads(
        self,
        *,
        pool: ThreadPoolExecutor,
        futures: dict[Future[str], str],
        download_workers: int,
    ) -> None:
        in_flight_cache_keys = set(futures.values())
        staged_or_in_flight = _count_staged_crpacks(self.staging_root, self.split) + len(futures)
        self.active_order = [
            cache_key for cache_key in self.active_order if cache_key in self.active_owner_by_key
        ]
        for cache_key in self.active_order:
            if len(futures) >= download_workers:
                break
            if staged_or_in_flight >= self.max_staged_blocks:
                break
            if cache_key in in_flight_cache_keys:
                continue
            if v15_block_is_cached(self.staging_root, self.split, cache_key):
                continue
            futures[
                pool.submit(
                    _download_and_stage_crpack,
                    staging_source_root=self.staging_root,
                    split=self.split,
                    cache_key=cache_key,
                    relative_path=self.block_paths_by_key[cache_key],
                )
            ] = cache_key
            in_flight_cache_keys.add(cache_key)
            staged_or_in_flight += 1

    def finish_download(self, future: Future[str], futures: dict[Future[str], str]) -> None:
        cache_key = futures.pop(future)
        future.result()
        if cache_key in self.released_cache_keys:
            clear_v15_block(self.staging_root, self.split, cache_key)


def _run_hf_v2_staging_downloader(
    *,
    split: str,
    staging_source_root: str,
    block_paths_by_key: dict[str, str],
    worker_queues: tuple[tuple[str, ...], ...],
    worker_quotas: tuple[int, ...],
    release_queue: Any,
    max_staged_blocks: int,
    download_workers: int,
) -> None:
    staging_root_path = Path(staging_source_root)
    try:
        runtime = _WorkerStagingRuntime(
            split=split,
            staging_root=staging_root_path,
            block_paths_by_key=block_paths_by_key,
            worker_queues=worker_queues,
            worker_quotas=worker_quotas,
            release_queue=release_queue,
            max_staged_blocks=max_staged_blocks,
        )
        runtime.fill_worker_slots()

        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            futures: dict[Future[str], str] = {}
            while futures or runtime.has_pending_work():
                runtime.drain_released_blocks()
                runtime.fill_worker_slots()
                runtime.submit_downloads(
                    pool=pool,
                    futures=futures,
                    download_workers=download_workers,
                )

                if not futures:
                    time.sleep(_HF_V2_STAGING_POLL_SECONDS)
                    continue

                done, _pending = wait(
                    futures,
                    timeout=_HF_V2_STAGING_POLL_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    runtime.finish_download(future, futures)
    except BaseException as exc:
        _write_staging_error(staging_root_path, split, exc)
        raise


class HFV2StagedBlockReader:
    requires_dataloader_prepare = True

    def __init__(
        self,
        *,
        split: str,
        staging_root: Path | None = None,
        max_staged_blocks: int = HF_V2_STAGING_MAX_BLOCKS,
        download_workers: int | None = None,
    ) -> None:
        self.split = split
        self.staging_root = Path(staging_root) if staging_root is not None else resolve_streaming_stage_root()
        self.max_staged_blocks = max(1, int(max_staged_blocks))
        self.download_workers = resolve_hf_v2_download_workers(download_workers)
        stage_key = str(abs(hash((HF_V2_DATASET_ID, split, os.getpid(), id(self)))))
        self.staging_source_root = self.staging_root / stage_key
        self.block_path_by_key: dict[str, str] = {}
        self.row_count_by_key: dict[str, int] = {}
        self.worker_by_key: dict[str, int] = {}
        self.worker_quotas: tuple[int, ...] = ()
        self.effective_max_staged_blocks = self.max_staged_blocks
        self._release_queue: Any | None = None
        self._process: mp.Process | None = None
        self._process_owner_pid: int | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_process"] = None
        state["_process_owner_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def prepare_blocks(self, blocks: tuple[dict[str, Any], ...], *, worker_count: int = 1) -> None:
        self.block_path_by_key = _block_path_by_key(blocks)
        self.row_count_by_key = _row_count_by_key(blocks)
        cache_keys = tuple(str(block["cache_key"]) for block in blocks)
        staging_plan = _build_worker_staging_plan(
            cache_keys,
            max_staged_blocks=self.max_staged_blocks,
            worker_count=worker_count,
        )
        self.worker_by_key = dict(staging_plan.worker_by_key)
        self.worker_quotas = staging_plan.worker_quotas
        self.effective_max_staged_blocks = staging_plan.effective_max_staged_blocks
        remove_tree(self.staging_source_root)
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        context = mp.get_context("spawn")
        self._release_queue = context.Queue()
        self._process = context.Process(
            target=_run_hf_v2_staging_downloader,
            kwargs={
                "split": self.split,
                "staging_source_root": os.fspath(self.staging_source_root),
                "block_paths_by_key": dict(self.block_path_by_key),
                "worker_queues": staging_plan.worker_queues,
                "worker_quotas": staging_plan.worker_quotas,
                "release_queue": self._release_queue,
                "max_staged_blocks": staging_plan.effective_max_staged_blocks,
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
                raise RuntimeError(f"HF v2 streaming downloader exited with code {process.exitcode}")
            time.sleep(_HF_V2_STAGING_POLL_SECONDS)
        return load_v15_block(self.staging_source_root, self.split, cache_key)

    def block_is_ready(self, cache_key: str) -> bool:
        _raise_if_staging_error(self.staging_source_root, self.split)
        process = self._process if self._process_owner_pid == os.getpid() else None
        if process is not None and not process.is_alive() and process.exitcode not in (None, 0):
            raise RuntimeError(f"HF v2 streaming downloader exited with code {process.exitcode}")
        return v15_block_is_cached(self.staging_source_root, self.split, cache_key)

    def release_block(self, cache_key: str) -> None:
        clear_v15_block(self.staging_source_root, self.split, cache_key)
        if self._release_queue is not None:
            self._release_queue.put(cache_key)

    def close(self) -> None:
        process = self._process if self._process_owner_pid == os.getpid() else None
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if self._process_owner_pid == os.getpid():
            remove_tree(self.staging_source_root)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "BlockReader",
    "HFV2LocalBlockReader",
    "HFV2StagedBlockReader",
    "HF_V2_DATASET_ID",
    "HF_V2_DOWNLOAD_WORKERS",
    "HF_V2_LAYOUT",
    "HF_V2_REVISION",
    "HF_V2_STAGING_MAX_BLOCKS",
    "ensure_hf_v2_local_blocks",
    "hf_v2_download_worker_count",
    "is_hf_v2_full_split_complete",
    "load_hf_v2_manifest",
    "load_hf_v2_split_catalog",
    "normalize_hf_v2_catalog",
    "resolve_dataset_root",
    "resolve_hf_v2_download_workers",
    "resolve_streaming_stage_root",
]
