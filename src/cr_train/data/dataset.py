from __future__ import annotations

import hashlib
import inspect
import math
import os
import random
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset, get_worker_info

from .hf_v2 import (
    BlockReader,
    HFV2LocalBlockReader,
    HFV2StagedBlockReader,
    ensure_hf_v2_local_blocks,
    load_hf_v2_manifest,
    load_hf_v2_split_catalog,
)
from .constants import OPTICAL_CHANNELS, SAR_CHANNELS
from .planning import plan_sample
from .runtime import (
    _render_warmup_timeline,
    emit_startup_event,
    get_rank,
    get_world_size,
    run_startup_stage,
)
from .store import as_bytes

_TRAIN_ACTIVE_BLOCK_COUNT = 8
_ROW_START_KEY = "_cr_train_row_start"
_ROW_STOP_KEY = "_cr_train_row_stop"


def _call_prepare_blocks(
    prepare_blocks,
    blocks: tuple[dict[str, Any], ...],
    *,
    worker_count: int,
    worker_blocks: tuple[tuple[dict[str, Any], ...], ...] | None = None,
) -> None:
    signature = inspect.signature(prepare_blocks)
    parameters = signature.parameters
    accepts_worker_count = "worker_count" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    accepts_worker_blocks = "worker_blocks" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if accepts_worker_count:
        kwargs["worker_count"] = worker_count
    if accepts_worker_blocks:
        kwargs["worker_blocks"] = worker_blocks
    prepare_blocks(blocks, **kwargs)


@dataclass(slots=True)
class PreparedSplit:
    """DataLoader-ready split backed by HF v2 blocks."""

    dataset: TorchIterableDataset[dict[str, Any]]
    num_examples: int


@dataclass(slots=True)
class PreparedSplitState:
    """Static split selection resolved against HF v2 blocks."""

    split: str
    block_reader: BlockReader
    streaming: bool
    source_stage: str
    seed: int
    requested_rows: int
    effective_rows: int
    required_blocks: int
    planner_mode: str
    selected_blocks: tuple[dict[str, Any], ...]
    row_counts_by_key: dict[str, int]


@dataclass(slots=True, frozen=True)
class _SpatialTransformParams:
    top: int
    left: int
    height: int
    width: int
    flip_vertical: bool = False
    flip_horizontal: bool = False
    rot90_k: int = 0


@dataclass(slots=True, eq=False)
class _ActiveBlockCursor:
    cache_key: str
    rows: Any
    indices: list[int]
    next_index: int = 0

    def pop(self) -> dict[str, Any]:
        row = self.rows[self.indices[self.next_index]]
        self.next_index += 1
        return row

    @property
    def exhausted(self) -> bool:
        return self.next_index >= len(self.indices)


@dataclass(slots=True)
class _BatchCollateFn:
    include_metadata: bool = True
    crop_size: int | None = None
    crop_mode: str = "none"
    random_flip: bool = False
    random_rot90: bool = False

    def __post_init__(self) -> None:
        self.include_metadata = bool(self.include_metadata)
        self.crop_mode = _normalize_crop_mode(self.crop_mode)
        self.random_flip = bool(self.random_flip)
        self.random_rot90 = bool(self.random_rot90)
        if self.crop_size is not None and self.crop_size <= 0:
            raise ValueError("crop_size must be greater than zero when provided")
        if self.crop_mode != "none" and self.crop_size is None:
            raise ValueError("crop_size must be provided when crop_mode is not 'none'")

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty batch")

        first = rows[0]
        sar_chw = _resolve_chw_shape(first["sar_shape"], SAR_CHANNELS)
        opt_chw = _resolve_chw_shape(first["opt_shape"], OPTICAL_CHANNELS)
        transformed_sar_chw = _resolve_transformed_shape(
            sar_chw,
            crop_size=self.crop_size,
            crop_mode=self.crop_mode,
        )
        transformed_opt_chw = _resolve_transformed_shape(
            opt_chw,
            crop_size=self.crop_size,
            crop_mode=self.crop_mode,
        )
        use_spatial_transform = _has_spatial_transform(
            crop_mode=self.crop_mode,
            random_flip=self.random_flip,
            random_rot90=self.random_rot90,
        )
        if use_spatial_transform and sar_chw[1:] != opt_chw[1:]:
            raise ValueError(
                "spatial transforms require matching SAR and optical spatial dimensions"
            )

        batch_size = len(rows)
        sar_batch = torch.empty((batch_size, *transformed_sar_chw), dtype=torch.float32)
        cloudy_batch = torch.empty((batch_size, *transformed_opt_chw), dtype=torch.float32)
        target_batch = torch.empty((batch_size, *transformed_opt_chw), dtype=torch.float32)

        metadata = {"season": [], "scene": [], "patch": []} if self.include_metadata else None
        for i, row in enumerate(rows):
            if use_spatial_transform:
                params = _sample_spatial_transform_params(
                    height=opt_chw[1],
                    width=opt_chw[2],
                    crop_size=self.crop_size,
                    crop_mode=self.crop_mode,
                    random_flip=self.random_flip,
                    random_rot90=self.random_rot90,
                )

                sar_image = torch.empty(sar_chw, dtype=torch.float32)
                _decode_image_into(
                    sar_image,
                    row["sar"],
                    row["sar_shape"],
                    src_dtype=torch.float32,
                    expected_channels=SAR_CHANNELS,
                )
                _fill_nan_tensor(sar_image)
                _normalize_sar_tensor(sar_image)
                _assert_finite_tensor(sar_image, field="sar", row=row)
                sar_transformed = _apply_spatial_transform(sar_image, params)
                sar_batch[i].copy_(sar_transformed)

                cloudy_image = torch.empty(opt_chw, dtype=torch.float32)
                _decode_image_into(
                    cloudy_image,
                    row["cloudy"],
                    row["opt_shape"],
                    src_dtype=torch.int16,
                    expected_channels=OPTICAL_CHANNELS,
                )
                _fill_nan_tensor(cloudy_image)
                _normalize_optical_tensor(cloudy_image)
                _assert_finite_tensor(cloudy_image, field="cloudy", row=row)
                cloudy_transformed = _apply_spatial_transform(cloudy_image, params)
                cloudy_batch[i].copy_(cloudy_transformed)

                target_image = torch.empty(opt_chw, dtype=torch.float32)
                _decode_image_into(
                    target_image,
                    row["target"],
                    row["opt_shape"],
                    src_dtype=torch.int16,
                    expected_channels=OPTICAL_CHANNELS,
                )
                _fill_nan_tensor(target_image)
                _normalize_optical_tensor(target_image)
                _assert_finite_tensor(target_image, field="target", row=row)
                target_transformed = _apply_spatial_transform(target_image, params)
                target_batch[i].copy_(target_transformed)
            else:
                _decode_image_into(
                    sar_batch[i],
                    row["sar"],
                    row["sar_shape"],
                    src_dtype=torch.float32,
                    expected_channels=SAR_CHANNELS,
                )
                _fill_nan_tensor(sar_batch[i])
                _normalize_sar_tensor(sar_batch[i])
                _assert_finite_tensor(sar_batch[i], field="sar", row=row)
                _decode_image_into(
                    cloudy_batch[i],
                    row["cloudy"],
                    row["opt_shape"],
                    src_dtype=torch.int16,
                    expected_channels=OPTICAL_CHANNELS,
                )
                _fill_nan_tensor(cloudy_batch[i])
                _normalize_optical_tensor(cloudy_batch[i])
                _assert_finite_tensor(cloudy_batch[i], field="cloudy", row=row)
                _decode_image_into(
                    target_batch[i],
                    row["target"],
                    row["opt_shape"],
                    src_dtype=torch.int16,
                    expected_channels=OPTICAL_CHANNELS,
                )
                _fill_nan_tensor(target_batch[i])
                _normalize_optical_tensor(target_batch[i])
                _assert_finite_tensor(target_batch[i], field="target", row=row)
            if metadata is not None:
                metadata["season"].append(str(row.get("season", "")))
                metadata["scene"].append(str(row.get("scene", "")))
                metadata["patch"].append(str(row.get("patch", "")))

        batch: dict[str, Any] = {"sar": sar_batch, "cloudy": cloudy_batch, "target": target_batch}
        if metadata is not None:
            batch["meta"] = metadata
        return batch


def resolve_num_workers(num_workers: int | str) -> int:
    """Resolve DataLoader worker count from an int or ``'auto'``."""
    if isinstance(num_workers, int):
        return max(0, num_workers)
    if num_workers != "auto":
        raise ValueError("num_workers must be an integer or 'auto'")
    cpu_count = os.cpu_count() or 1
    return min(16, max(1, cpu_count // 3))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _derive_named_seed(seed: int, split: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{purpose}:{split}".encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "big")


def _derive_block_seed(seed: int, *, split: str, epoch: int, cache_key: str) -> int:
    digest = hashlib.sha256(f"{split}:{epoch}:{cache_key}".encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "big")


def _derive_worker_seed(seed: int, *, split: str, epoch: int, worker_id: int) -> int:
    digest = hashlib.sha256(f"{split}:{epoch}:worker:{worker_id}".encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "big")


def _shuffle_blocks(
    blocks: list[dict[str, Any]],
    *,
    seed: int,
    split: str,
    epoch: int,
) -> list[dict[str, Any]]:
    if len(blocks) <= 1:
        return list(blocks)
    rng = np.random.default_rng(_derive_named_seed(seed + epoch, split, "epoch-block-order"))
    order = rng.permutation(len(blocks))
    return [blocks[int(index)] for index in order.tolist()]


def _block_row_range(block: dict[str, Any], *, row_count: int) -> tuple[int, int]:
    row_start = int(block.get(_ROW_START_KEY, 0))
    row_stop = int(block.get(_ROW_STOP_KEY, row_count))
    if row_start < 0 or row_stop < row_start or row_stop > row_count:
        raise ValueError(
            f"invalid block row range for {block.get('cache_key')!r}: "
            f"{row_start}:{row_stop} outside 0:{row_count}"
        )
    return row_start, row_stop


def _copy_block_with_row_range(
    block: dict[str, Any],
    *,
    row_start: int,
    row_stop: int,
    row_count: int,
) -> dict[str, Any]:
    sliced = dict(block)
    if row_start == 0 and row_stop == row_count:
        sliced.pop(_ROW_START_KEY, None)
        sliced.pop(_ROW_STOP_KEY, None)
        return sliced
    sliced[_ROW_START_KEY] = row_start
    sliced[_ROW_STOP_KEY] = row_stop
    return sliced


def _block_row_count_from_state(state: PreparedSplitState, block: dict[str, Any]) -> int:
    source_row_count = int(state.row_counts_by_key[str(block["cache_key"])])
    row_start, row_stop = _block_row_range(block, row_count=source_row_count)
    return row_stop - row_start


def _block_row_count_from_metadata(block: dict[str, Any]) -> int | None:
    if "row_count" not in block:
        return None
    source_row_count = int(block["row_count"])
    row_start, row_stop = _block_row_range(block, row_count=source_row_count)
    return row_stop - row_start


def _slice_blocks_by_row_offsets(
    blocks: list[dict[str, Any]],
    *,
    row_start: int,
    row_stop: int,
    source_row_count_for_block: Callable[[dict[str, Any]], int],
) -> list[dict[str, Any]]:
    sliced_blocks: list[dict[str, Any]] = []
    row_offset = 0
    for block in blocks:
        source_row_count = source_row_count_for_block(block)
        source_start, source_stop = _block_row_range(block, row_count=source_row_count)
        row_count = source_stop - source_start
        block_start = row_offset
        block_stop = block_start + row_count
        row_offset = block_stop

        overlap_start = max(row_start, block_start)
        overlap_stop = min(row_stop, block_stop)
        if overlap_start >= overlap_stop:
            continue

        sliced_blocks.append(
            _copy_block_with_row_range(
                block,
                row_start=source_start + (overlap_start - block_start),
                row_stop=source_start + (overlap_stop - block_start),
                row_count=source_row_count,
            )
        )
    return sliced_blocks


def _slice_blocks_for_rank(
    state: PreparedSplitState,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    world_size = get_world_size()
    if world_size <= 1:
        return list(blocks)

    total_rows = _count_rows_from_state(state, blocks)
    rows_per_rank = total_rows // world_size
    if rows_per_rank <= 0:
        raise RuntimeError(
            f"split {state.split!r} has {total_rows} rows, which is too small for "
            f"world_size={world_size}"
        )

    rank = get_rank()
    rank_start = rank * rows_per_rank
    rank_stop = rank_start + rows_per_rank
    return _slice_blocks_by_row_offsets(
        blocks,
        row_start=rank_start,
        row_stop=rank_stop,
        source_row_count_for_block=lambda block: int(state.row_counts_by_key[str(block["cache_key"])]),
    )


def _slice_blocks_for_worker(
    blocks: list[dict[str, Any]],
    *,
    worker_id: int,
    worker_count: int,
) -> list[dict[str, Any]]:
    if worker_count <= 1:
        return list(blocks)

    total_rows = 0
    for block in blocks:
        row_count = _block_row_count_from_metadata(block)
        if row_count is None:
            return [block for index, block in enumerate(blocks) if index % worker_count == worker_id]
        total_rows += row_count

    rows_per_worker, remainder = divmod(total_rows, worker_count)
    worker_start = worker_id * rows_per_worker + min(worker_id, remainder)
    worker_rows = rows_per_worker + (1 if worker_id < remainder else 0)
    return _slice_blocks_by_row_offsets(
        blocks,
        row_start=worker_start,
        row_stop=worker_start + worker_rows,
        source_row_count_for_block=lambda block: int(block["row_count"]),
    )


def _count_rows_from_state(state: PreparedSplitState, blocks: list[dict[str, Any]]) -> int:
    total = 0
    for block in blocks:
        total += _block_row_count_from_state(state, block)
    return total


class BlockIterableDataset(TorchIterableDataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        block_reader: BlockReader,
        blocks: tuple[dict[str, Any], ...],
        seed: int,
        epoch: int,
        split: str,
        training: bool,
    ) -> None:
        self.block_reader = block_reader
        self.blocks = blocks
        self.seed = seed
        self.epoch = epoch
        self.split = split
        self.training = training
        self._prepared_for_dataloader = False

    def prepare_for_dataloader(self, *, num_workers: int) -> None:
        prepare_blocks = getattr(self.block_reader, "prepare_blocks", None)
        if prepare_blocks is not None:
            worker_count = max(1, int(num_workers))
            worker_blocks = tuple(
                tuple(
                    _slice_blocks_for_worker(
                        list(self.blocks),
                        worker_id=worker_id,
                        worker_count=worker_count,
                    )
                )
                for worker_id in range(worker_count)
            )
            _call_prepare_blocks(
                prepare_blocks,
                tuple(self.blocks),
                worker_count=worker_count,
                worker_blocks=worker_blocks,
            )
        self._prepared_for_dataloader = True

    def _load_block_cursor(
        self,
        *,
        block: dict[str, Any],
        shuffle_rows: bool,
    ) -> _ActiveBlockCursor:
        cache_key = str(block["cache_key"])
        rows = self.block_reader.load_block(cache_key)
        row_start, row_stop = _block_row_range(block, row_count=len(rows))
        indices = list(range(row_start, row_stop))
        if shuffle_rows and len(indices) > 1:
            row_rng = random.Random(
                _derive_block_seed(
                    self.seed,
                    split=self.split,
                    epoch=self.epoch,
                    cache_key=cache_key,
                )
            )
            row_rng.shuffle(indices)
        return _ActiveBlockCursor(cache_key=cache_key, rows=rows, indices=indices)

    def _load_active_block(self, *, block: dict[str, Any]) -> _ActiveBlockCursor:
        return self._load_block_cursor(block=block, shuffle_rows=True)

    def _load_block_rows(self, *, block: dict[str, Any]):
        cache_key = str(block["cache_key"])
        rows = self.block_reader.load_block(cache_key)
        row_start, row_stop = _block_row_range(block, row_count=len(rows))
        if row_start == 0 and row_stop == len(rows):
            return rows
        return rows[row_start:row_stop]

    def _prefetch_blocks(self) -> bool:
        return bool(getattr(self.block_reader, "prefetch_blocks", False))

    def _release_block(self, cache_key: str, *, worker_id: int | None = None) -> None:
        release_block = getattr(self.block_reader, "release_block", None)
        if release_block is not None:
            signature = inspect.signature(release_block)
            parameters = signature.parameters
            accepts_worker_id = "worker_id" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if accepts_worker_id:
                release_block(cache_key, worker_id=worker_id)
            else:
                release_block(cache_key)

    def _block_is_ready(self, cache_key: str) -> bool:
        block_is_ready = getattr(self.block_reader, "block_is_ready", None)
        if block_is_ready is None:
            return True
        return bool(block_is_ready(cache_key))

    def _close_reader(self) -> None:
        close = getattr(self.block_reader, "close", None)
        if close is not None:
            close()

    def _iter_training_rows(self, *, blocks: list[dict[str, Any]], worker_id: int):
        if not blocks:
            return

        mix_rng = random.Random(
            _derive_worker_seed(
                self.seed,
                split=self.split,
                epoch=self.epoch,
                worker_id=worker_id,
            )
        )
        active_blocks: list[_ActiveBlockCursor] = []
        round_robin: list[_ActiveBlockCursor] = []
        next_block_index = 0
        prefetch_pool: ThreadPoolExecutor | None = None
        prefetch_future: Future[_ActiveBlockCursor] | None = None

        def start_prefetch() -> None:
            nonlocal next_block_index, prefetch_pool, prefetch_future
            if not self._prefetch_blocks() or prefetch_future is not None:
                return
            if next_block_index >= len(blocks):
                return
            if prefetch_pool is None:
                prefetch_pool = ThreadPoolExecutor(max_workers=1)
            block = blocks[next_block_index]
            next_block_index += 1
            prefetch_future = prefetch_pool.submit(
                self._load_block_cursor,
                block=block,
                shuffle_rows=True,
            )

        def load_next_block(*, wait: bool) -> _ActiveBlockCursor | None:
            nonlocal next_block_index, prefetch_future
            if prefetch_future is not None:
                if not wait and not prefetch_future.done():
                    return None
                cursor = prefetch_future.result()
                prefetch_future = None
                return cursor
            if next_block_index >= len(blocks):
                return None
            block = blocks[next_block_index]
            cache_key = str(block["cache_key"])
            if not wait and not self._block_is_ready(cache_key):
                return None
            next_block_index += 1
            return self._load_active_block(block=block)

        def refill_active_blocks(*, target_count: int, wait: bool, max_new_blocks: int | None = None) -> None:
            added_blocks = 0
            while len(active_blocks) < target_count:
                if max_new_blocks is not None and added_blocks >= max_new_blocks:
                    break
                cursor = load_next_block(wait=wait)
                if cursor is None:
                    break
                active_blocks.append(cursor)
                added_blocks += 1
            start_prefetch()

        try:
            refill_active_blocks(target_count=1, wait=True)
            while active_blocks:
                if not round_robin:
                    round_robin = list(active_blocks)
                    mix_rng.shuffle(round_robin)

                current_block = round_robin.pop()
                yield current_block.pop()
                refill_active_blocks(
                    target_count=_TRAIN_ACTIVE_BLOCK_COUNT,
                    wait=False,
                    max_new_blocks=1,
                )

                if current_block.exhausted:
                    cache_key = current_block.cache_key
                    active_blocks.remove(current_block)
                    round_robin = [
                        candidate for candidate in round_robin if candidate is not current_block
                    ]
                    del current_block
                    self._release_block(cache_key, worker_id=worker_id)
                    if not active_blocks:
                        refill_active_blocks(target_count=1, wait=True)
        finally:
            if prefetch_future is not None and prefetch_future.done():
                try:
                    prefetched_cursor = prefetch_future.result()
                except Exception:
                    pass
                else:
                    self._release_block(prefetched_cursor.cache_key, worker_id=worker_id)
            if prefetch_future is not None:
                prefetch_future.cancel()
            if prefetch_pool is not None:
                prefetch_pool.shutdown(wait=False, cancel_futures=True)
            for cursor in active_blocks:
                self._release_block(cursor.cache_key, worker_id=worker_id)
            self._close_reader()

    def _iter_evaluation_rows(self, *, blocks: list[dict[str, Any]], worker_id: int):
        if not self._prefetch_blocks() or len(blocks) <= 1:
            try:
                for block in blocks:
                    cache_key = str(block["cache_key"])
                    rows = self._load_block_rows(block=block)
                    try:
                        yield from rows
                    finally:
                        del rows
                        self._release_block(cache_key, worker_id=worker_id)
            finally:
                self._close_reader()
            return

        prefetch_pool = ThreadPoolExecutor(max_workers=1)
        prefetch_future: Future[Any] | None = None
        try:
            for index, block in enumerate(blocks):
                if prefetch_future is None:
                    rows = self._load_block_rows(block=block)
                else:
                    rows = prefetch_future.result()
                    prefetch_future = None
                next_index = index + 1
                if next_index < len(blocks):
                    prefetch_future = prefetch_pool.submit(
                        self._load_block_rows,
                        block=blocks[next_index],
                    )
                cache_key = str(block["cache_key"])
                try:
                    yield from rows
                finally:
                    del rows
                    self._release_block(cache_key, worker_id=worker_id)
        finally:
            if prefetch_future is not None and prefetch_future.done():
                try:
                    prefetched_rows = prefetch_future.result()
                except Exception:
                    pass
                else:
                    del prefetched_rows
                    next_index = index + 1
                    if next_index < len(blocks):
                        self._release_block(str(blocks[next_index]["cache_key"]), worker_id=worker_id)
            if prefetch_future is not None:
                prefetch_future.cancel()
            prefetch_pool.shutdown(wait=False, cancel_futures=True)
            self._close_reader()

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None and not self._prepared_for_dataloader:
            self.prepare_for_dataloader(num_workers=0)
        elif worker_info is not None and not self._prepared_for_dataloader:
            if getattr(self.block_reader, "requires_dataloader_prepare", False):
                raise RuntimeError("BlockIterableDataset must be prepared before worker iteration")
            self._prepared_for_dataloader = True
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1
        blocks = _slice_blocks_for_worker(
            list(self.blocks),
            worker_id=worker_id,
            worker_count=worker_count,
        )

        if self.training:
            yield from self._iter_training_rows(blocks=blocks, worker_id=worker_id)
            return

        yield from self._iter_evaluation_rows(blocks=blocks, worker_id=worker_id)


def _resolve_selected_blocks(
    catalog: dict[str, Any],
    *,
    selected_indices: list[int],
) -> list[dict[str, Any]]:
    catalog_blocks = list(catalog.get("blocks", []))
    return [catalog_blocks[index] for index in selected_indices]


def _resolve_reader_selected_block_row_counts(
    block_reader: Any,
    selected_blocks: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, int]:
    cache_keys = tuple(str(block["cache_key"]) for block in selected_blocks)
    load_many = getattr(block_reader, "load_block_metadata_many", None)
    metadata_by_key = (
        load_many(cache_keys)
        if load_many is not None
        else {
            cache_key: block_reader.load_block_metadata(cache_key)
            for cache_key in cache_keys
        }
    )
    missing_cache_keys: list[str] = []
    row_counts_by_key: dict[str, int] = {}
    for block in selected_blocks:
        cache_key = str(block["cache_key"])
        metadata = metadata_by_key.get(cache_key)
        if metadata is None:
            missing_cache_keys.append(cache_key)
            continue
        row_count = int(metadata["row_count"])
        expected_row_count = int(block.get("row_count", row_count))
        if row_count != expected_row_count:
            missing_cache_keys.append(cache_key)
            continue
        row_counts_by_key[cache_key] = row_count
    if missing_cache_keys:
        raise FileNotFoundError(f"{source_name} split is missing requested blocks: {', '.join(missing_cache_keys)}")
    return row_counts_by_key


def _resolve_catalog_selected_block_row_counts(
    selected_blocks: list[dict[str, Any]],
) -> dict[str, int]:
    row_counts_by_key: dict[str, int] = {}
    for block in selected_blocks:
        cache_key = str(block["cache_key"])
        row_counts_by_key[cache_key] = int(block["row_count"])
    return row_counts_by_key


def _emit_streaming_selection_ready(
    startup_callback,
    *,
    split: str,
    sample_plan,
    selected_blocks: list[dict[str, Any]],
) -> None:
    event = {
        "stage": "warm streaming data",
        "split": split,
        "requested_rows": sample_plan.requested_rows,
        "effective_rows": sum(int(block["row_count"]) for block in selected_blocks),
        "required_blocks": sample_plan.required_blocks,
        "planner_mode": sample_plan.planner_mode,
        "selected_block_count": sample_plan.required_blocks,
        "ready_selected_blocks": sample_plan.required_blocks,
        "selected_missing_blocks": 0,
        "execution_block_count": int(sample_plan.execution_block_count),
        "resolved_blocks": 0,
    }
    emit_startup_event(startup_callback, status="start", **event)
    emit_startup_event(
        startup_callback,
        status="done",
        elapsed_sec=0.0,
        timeline=_render_warmup_timeline(
            sample_plan.selected_bitmap,
            stop_block=sample_plan.execution_block_count,
        ),
        **event,
    )


def resolve_prepared_split_state(
    *,
    split: str,
    dataset_name: str,
    revision: str | None,
    max_samples: int | None,
    seed: int,
    dataset_root: Path | None = None,
    streaming: bool = True,
    startup_callback=None,
) -> PreparedSplitState:
    """Resolve the static block selection for a split from the HF v2 block layout."""
    del dataset_name, revision
    load_hf_v2_manifest(dataset_root=dataset_root, streaming=streaming)
    catalog = load_hf_v2_split_catalog(
        split=split,
        dataset_root=dataset_root,
        streaming=streaming,
    )

    sample_plan = plan_sample(
        catalog,
        seed,
        max_samples,
        split=split,
    )
    selected_blocks = _resolve_selected_blocks(
        catalog,
        selected_indices=[int(index) for index in sample_plan.selected_blocks.tolist()],
    )
    row_counts_by_key = _resolve_catalog_selected_block_row_counts(selected_blocks)
    if streaming:
        block_reader: BlockReader = HFV2StagedBlockReader(split=split)
        source_stage = "load streaming data"
        _emit_streaming_selection_ready(
            startup_callback,
            split=split,
            sample_plan=sample_plan,
            selected_blocks=selected_blocks,
        )
    else:
        if dataset_root is None:
            raise ValueError("dataset_root must be provided when streaming=False")
        ensure_hf_v2_local_blocks(
            dataset_root=dataset_root,
            split=split,
            catalog=catalog,
            selected_blocks=tuple(selected_blocks),
            requested_rows=sample_plan.requested_rows,
            effective_rows=int(sum(row_counts_by_key[str(block["cache_key"])] for block in selected_blocks)),
            required_blocks=sample_plan.required_blocks,
            planner_mode=sample_plan.planner_mode,
            execution_block_count=sample_plan.execution_block_count,
            full_split=sample_plan.planner_mode == "full_split",
            timeline=_render_warmup_timeline(
                sample_plan.selected_bitmap,
                stop_block=sample_plan.execution_block_count,
            ),
            startup_callback=startup_callback,
        )
        block_reader = HFV2LocalBlockReader(
            dataset_root=dataset_root,
            block_path_by_key={str(block["cache_key"]): str(block["path"]) for block in selected_blocks},
            row_count_by_key=row_counts_by_key,
        )
        source_stage = "load local data"
    return PreparedSplitState(
        split=split,
        block_reader=block_reader,
        streaming=streaming,
        source_stage=source_stage,
        seed=seed,
        requested_rows=sample_plan.requested_rows,
        effective_rows=int(sum(row_counts_by_key[str(block["cache_key"])] for block in selected_blocks)),
        required_blocks=sample_plan.required_blocks,
        planner_mode=sample_plan.planner_mode,
        selected_blocks=tuple(selected_blocks),
        row_counts_by_key=row_counts_by_key,
    )


def prepare_split_from_state(
    state: PreparedSplitState,
    *,
    epoch: int,
    training: bool,
    startup_callback=None,
) -> PreparedSplit:
    """Build a PreparedSplit from a pre-resolved split state."""
    selected_blocks = list(state.selected_blocks)
    ordered_blocks = _shuffle_blocks(selected_blocks, seed=state.seed, split=state.split, epoch=epoch) if training else selected_blocks
    rank_blocks = _slice_blocks_for_rank(state, ordered_blocks)
    num_examples = _count_rows_from_state(state, rank_blocks)
    dataset = run_startup_stage(
        startup_callback,
        stage=state.source_stage,
        split=state.split,
        operation=lambda: BlockIterableDataset(
            block_reader=state.block_reader,
            blocks=tuple(rank_blocks),
            seed=state.seed,
            epoch=epoch,
            split=state.split,
            training=training,
        ),
        requested_rows=state.requested_rows,
        effective_rows=state.effective_rows,
        required_blocks=state.required_blocks,
        planner_mode=state.planner_mode,
    )
    return PreparedSplit(dataset=dataset, num_examples=num_examples)


def prepare_split(
    *,
    split: str,
    dataset_name: str,
    revision: str | None,
    max_samples: int | None,
    seed: int,
    epoch: int,
    training: bool,
    dataset_root: Path | None = None,
    streaming: bool = True,
    startup_callback=None,
) -> PreparedSplit:
    """Build a PreparedSplit from the HF v2 block dataset."""
    state = resolve_prepared_split_state(
        split=split,
        dataset_name=dataset_name,
        revision=revision,
        max_samples=max_samples,
        seed=seed,
        dataset_root=dataset_root,
        streaming=streaming,
        startup_callback=startup_callback,
    )
    return prepare_split_from_state(
        state,
        epoch=epoch,
        training=training,
        startup_callback=startup_callback,
    )


def _as_shape(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"shape must be a list or tuple, got {type(value)!r}")
    shape = tuple(int(dim) for dim in value)
    if len(shape) != 3:
        raise ValueError(f"expected a 3D tensor shape, got {shape!r}")
    return shape  # type: ignore[return-value]


def _decode_image(buffer: Any, shape: Any, *, dtype: np.dtype[Any], expected_channels: int) -> np.ndarray:
    if isinstance(buffer, np.ndarray):
        image = np.asarray(buffer)
    else:
        resolved_shape = _as_shape(shape)
        raw = np.frombuffer(as_bytes(buffer), dtype=dtype)
        expected_size = math.prod(resolved_shape)
        if raw.size != expected_size:
            raise ValueError(f"buffer size mismatch for shape {resolved_shape}: expected {expected_size}, got {raw.size}")
        image = raw.reshape(resolved_shape)

    if image.shape[-1] == expected_channels and image.shape[0] != expected_channels:
        chw = np.transpose(image, (2, 0, 1))
    elif image.shape[0] == expected_channels:
        chw = image
    else:
        raise ValueError(f"could not infer channel dimension from shape {image.shape!r}")
    return np.ascontiguousarray(chw, dtype=np.float32)


_SAR_NORMALIZATION = (
    (0, -25.0, 0.0, 25.0, 2.0 / 25.0),
    (1, -32.5, 0.0, 32.5, 2.0 / 32.5),
)
_OPTICAL_CLAMP_RANGE = (0.0, 10000.0)
_OPTICAL_SCALE = 1.0 / 2000.0


def _row_metadata_key(row: dict[str, Any] | None) -> tuple[str, str, str]:
    if row is None:
        return ("", "", "")
    return (
        str(row.get("season", "")),
        str(row.get("scene", "")),
        str(row.get("patch", "")),
    )


def _format_row_context(row: dict[str, Any] | None) -> str:
    season, scene, patch = _row_metadata_key(row)
    parts = []
    if season:
        parts.append(f"season={season}")
    if scene:
        parts.append(f"scene={scene}")
    if patch:
        parts.append(f"patch={patch}")
    if parts:
        return ", ".join(parts)
    return "metadata unavailable"


def _fill_nan_numpy(image: np.ndarray) -> None:
    nan_mask = np.isnan(image)
    if not bool(nan_mask.any()):
        return
    valid = image[~nan_mask]
    if valid.size == 0:
        return
    image[nan_mask] = valid.mean(dtype=np.float64)


def _fill_nan_tensor(image: torch.Tensor) -> None:
    nan_mask = torch.isnan(image)
    if not bool(nan_mask.any()):
        return
    valid = image[~nan_mask]
    if valid.numel() == 0:
        return
    image[nan_mask] = valid.mean()


def _assert_finite_numpy(image: np.ndarray, *, field: str, row: dict[str, Any] | None) -> None:
    if bool(np.isfinite(image).all()):
        return
    raise FloatingPointError(
        f"non-finite {field} after normalization: {_format_row_context(row)}"
    )


def _assert_finite_tensor(image: torch.Tensor, *, field: str, row: dict[str, Any] | None) -> None:
    if bool(torch.isfinite(image).all()):
        return
    raise FloatingPointError(
        f"non-finite {field} after normalization: {_format_row_context(row)}"
    )


def _normalize_sar_numpy(sar: np.ndarray) -> None:
    for channel, clamp_min, clamp_max, offset, scale in _SAR_NORMALIZATION:
        np.clip(sar[channel], clamp_min, clamp_max, out=sar[channel])
        sar[channel] += offset
        sar[channel] *= scale


def _normalize_optical_numpy(image: np.ndarray) -> None:
    np.clip(image, *_OPTICAL_CLAMP_RANGE, out=image)
    image *= _OPTICAL_SCALE


def _normalize_sar_tensor(sar: torch.Tensor) -> None:
    for channel, clamp_min, clamp_max, offset, scale in _SAR_NORMALIZATION:
        sar[channel].clamp_(clamp_min, clamp_max).add_(offset).mul_(scale)


def _normalize_optical_tensor(image: torch.Tensor) -> None:
    image.clamp_(*_OPTICAL_CLAMP_RANGE).mul_(_OPTICAL_SCALE)


def _normalize_crop_mode(crop_mode: str) -> str:
    if not isinstance(crop_mode, str):
        raise TypeError("crop_mode must be a string")
    normalized = crop_mode.strip().lower()
    if normalized not in {"none", "random", "center"}:
        raise ValueError("crop_mode must be one of 'none', 'random', or 'center'")
    return normalized


def _resolve_transformed_shape(
    chw_shape: tuple[int, int, int],
    *,
    crop_size: int | None,
    crop_mode: str,
) -> tuple[int, int, int]:
    if crop_mode == "none":
        return chw_shape
    if crop_size is None:
        raise ValueError("crop_size must be provided when crop_mode is not 'none'")
    if crop_size <= 0:
        raise ValueError("crop_size must be greater than zero")
    _, height, width = chw_shape
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"crop_size {crop_size} exceeds input spatial size {(height, width)!r}"
        )
    return (chw_shape[0], crop_size, crop_size)


def _has_spatial_transform(*, crop_mode: str, random_flip: bool, random_rot90: bool) -> bool:
    return crop_mode != "none" or random_flip or random_rot90


def _sample_spatial_transform_params(
    *,
    height: int,
    width: int,
    crop_size: int | None,
    crop_mode: str,
    random_flip: bool,
    random_rot90: bool,
) -> _SpatialTransformParams:
    if crop_mode == "none":
        top = 0
        left = 0
        target_height = height
        target_width = width
    else:
        if crop_size is None:
            raise ValueError("crop_size must be provided when crop_mode is not 'none'")
        if crop_size <= 0:
            raise ValueError("crop_size must be greater than zero")
        if crop_size > height or crop_size > width:
            raise ValueError(
                f"crop_size {crop_size} exceeds input spatial size {(height, width)!r}"
            )
        if crop_mode == "center":
            top = (height - crop_size) // 2
            left = (width - crop_size) // 2
        else:
            top = random.randint(0, height - crop_size)
            left = random.randint(0, width - crop_size)
        target_height = crop_size
        target_width = crop_size

    return _SpatialTransformParams(
        top=top,
        left=left,
        height=target_height,
        width=target_width,
        flip_vertical=bool(random_flip and random.random() < 0.5),
        flip_horizontal=bool(random_flip and random.random() < 0.5),
        rot90_k=random.randrange(4) if random_rot90 else 0,
    )


def _apply_spatial_transform(image: torch.Tensor, params: _SpatialTransformParams) -> torch.Tensor:
    transformed = image[
        :,
        params.top:params.top + params.height,
        params.left:params.left + params.width,
    ]
    if params.flip_vertical:
        transformed = torch.flip(transformed, dims=(-2,))
    if params.flip_horizontal:
        transformed = torch.flip(transformed, dims=(-1,))
    if params.rot90_k:
        transformed = torch.rot90(transformed, k=params.rot90_k, dims=(-2, -1))
    return transformed


def decode_row(row: dict[str, Any], *, include_metadata: bool = True) -> dict[str, Any]:
    """Decode one block row into CHW float32 arrays."""
    sar = _decode_image(row["sar"], row["sar_shape"], dtype=np.float32, expected_channels=SAR_CHANNELS)
    _fill_nan_numpy(sar)
    _normalize_sar_numpy(sar)
    _assert_finite_numpy(sar, field="sar", row=row)
    cloudy = _decode_image(row["cloudy"], row["opt_shape"], dtype=np.int16, expected_channels=OPTICAL_CHANNELS)
    _fill_nan_numpy(cloudy)
    _normalize_optical_numpy(cloudy)
    _assert_finite_numpy(cloudy, field="cloudy", row=row)
    target = _decode_image(row["target"], row["opt_shape"], dtype=np.int16, expected_channels=OPTICAL_CHANNELS)
    _fill_nan_numpy(target)
    _normalize_optical_numpy(target)
    _assert_finite_numpy(target, field="target", row=row)
    decoded = {"sar": sar, "cloudy": cloudy, "target": target}
    if include_metadata:
        decoded["meta"] = {
            "season": str(row.get("season", "")),
            "scene": str(row.get("scene", "")),
            "patch": str(row.get("patch", "")),
        }
    return decoded


def _resolve_chw_shape(shape: Any, expected_channels: int) -> tuple[int, int, int]:
    resolved = _as_shape(shape)
    if resolved[-1] == expected_channels and resolved[0] != expected_channels:
        return (resolved[2], resolved[0], resolved[1])
    if resolved[0] == expected_channels:
        return resolved
    raise ValueError(f"could not infer channel dimension from shape {resolved!r}")


def _as_writable_buffer(buffer: Any) -> bytearray | memoryview:
    if isinstance(buffer, bytearray):
        return buffer
    if isinstance(buffer, memoryview) and not buffer.readonly:
        return buffer
    return bytearray(as_bytes(buffer))


def _decode_image_into(
    dest: torch.Tensor,
    buffer: Any,
    shape: Any,
    *,
    src_dtype: torch.dtype,
    expected_channels: int,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    scale: float = 1.0,
) -> None:
    if isinstance(buffer, np.ndarray):
        image = np.asarray(buffer)
        if image.shape[-1] == expected_channels and image.shape[0] != expected_channels:
            image = np.transpose(image, (2, 0, 1))
        elif image.shape[0] != expected_channels:
            raise ValueError(f"could not infer channel dimension from shape {image.shape!r}")
        np.copyto(dest.numpy(), image, casting="unsafe")
        if clamp_min is not None or clamp_max is not None:
            dest.clamp_(clamp_min, clamp_max)
        if scale != 1.0:
            dest.mul_(scale)
        return

    raw = torch.frombuffer(_as_writable_buffer(buffer), dtype=src_dtype)
    resolved_shape = _as_shape(shape)
    expected_size = math.prod(resolved_shape)
    if raw.numel() != expected_size:
        raise ValueError(
            f"buffer size mismatch for shape {resolved_shape}: "
            f"expected {expected_size}, got {raw.numel()}"
        )

    image = raw.reshape(resolved_shape)
    if image.shape[-1] == expected_channels and image.shape[0] != expected_channels:
        image = image.permute(2, 0, 1)
    elif image.shape[0] != expected_channels:
        raise ValueError(f"could not infer channel dimension from shape {image.shape!r}")

    dest.copy_(image)
    if clamp_min is not None or clamp_max is not None:
        dest.clamp_(clamp_min, clamp_max)
    if scale != 1.0:
        dest.mul_(scale)


def build_collate_fn(
    *,
    include_metadata: bool = True,
    crop_size: int | None = None,
    crop_mode: str = "none",
    random_flip: bool = False,
    random_rot90: bool = False,
):
    """Build the batch collate function used by DataLoader workers."""
    return _BatchCollateFn(
        include_metadata=include_metadata,
        crop_size=crop_size,
        crop_mode=crop_mode,
        random_flip=random_flip,
        random_rot90=random_rot90,
    )


def build_dataloader(
    prepared: PreparedSplit,
    *,
    batch_size: int,
    num_workers: int,
    training: bool,
    seed: int,
    epoch: int,
    include_metadata: bool = True,
    pin_memory: bool = True,
    multiprocessing_context: str | None = None,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    crop_size: int | None = None,
    crop_mode: str = "none",
    random_flip: bool = False,
    random_rot90: bool = False,
) -> DataLoader:
    """Create the split DataLoader for the block iterable dataset."""
    del seed, epoch
    prepare_for_dataloader = getattr(prepared.dataset, "prepare_for_dataloader", None)
    if prepare_for_dataloader is not None:
        prepare_for_dataloader(num_workers=num_workers)
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": build_collate_fn(
            include_metadata=include_metadata,
            crop_size=crop_size,
            crop_mode=crop_mode,
            random_flip=random_flip,
            random_rot90=random_rot90,
        ),
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "drop_last": drop_last if training else False,
    }
    if num_workers > 0:
        if multiprocessing_context is not None:
            dataloader_kwargs["multiprocessing_context"] = multiprocessing_context
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(prepared.dataset, **dataloader_kwargs)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


__all__ = [
    "BlockIterableDataset",
    "PreparedSplit",
    "PreparedSplitState",
    "build_collate_fn",
    "build_dataloader",
    "decode_row",
    "move_batch_to_device",
    "prepare_split",
    "prepare_split_from_state",
    "resolve_prepared_split_state",
    "resolve_num_workers",
    "seed_everything",
    "seed_worker",
]
