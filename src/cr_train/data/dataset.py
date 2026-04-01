from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset, DistributedSampler

from .constants import CANONICAL_SHUFFLE_BUFFER_SIZE, DATA_COLUMNS, OPTICAL_CHANNELS, SAR_CHANNELS
from .planning import compress_execution_runs, plan_sample
from .runtime import get_rank, get_world_size, is_distributed
from .source import ensure_source_root, ensure_split_catalog, run_startup_stage
from .store import SplitBlockCache, as_bytes, load_or_init_block_cache, resolve_block_cache_paths, resolve_dataset_seed


@dataclass(slots=True)
class PreparedSplit:
    """캐시된 블록 데이터로 구성된 DataLoader-ready split."""

    dataset: TorchDataset[dict[str, Any]]


def resolve_num_workers(num_workers: int | str) -> int:
    """DataLoader 워커 수 결정. 정수 또는 ``'auto'`` (CPU 코어 기반 자동 산출)."""
    if isinstance(num_workers, int):
        return max(0, num_workers)
    if num_workers != "auto":
        raise ValueError("num_workers must be an integer or 'auto'")
    cpu_count = os.cpu_count() or 1
    return min(4, max(1, cpu_count // 4))


def seed_everything(seed: int, deterministic: bool) -> None:
    """모든 RNG를 고정하여 재현성 보장. ``deterministic=True`` 시 CuDNN도 결정적 모드."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_cached_row_refs(
    execution_runs,
    cache: SplitBlockCache,
) -> tuple[np.ndarray, np.ndarray]:
    row_chunk_refs: list[np.ndarray] = []
    row_offset_refs: list[np.ndarray] = []

    # 연속 블록을 단일 청크 범위로 병합
    for run in execution_runs:
        if run.kind != "take_cached":
            continue
        block_index = run.start_block
        while block_index < run.stop_block:
            chunk_id = cache.chunk_ids[block_index]
            if chunk_id < 0:
                raise KeyError(f"cached run references an uncached block: {block_index}")
            start_offset = cache.block_offsets[block_index]
            stop_offset = start_offset + cache.block_row_counts[block_index]
            next_block = block_index + 1
            while next_block < run.stop_block:
                next_chunk_id = cache.chunk_ids[next_block]
                next_offset = cache.block_offsets[next_block]
                next_stop = next_offset + cache.block_row_counts[next_block]
                if next_chunk_id != chunk_id or next_offset != stop_offset:
                    break
                stop_offset = next_stop
                next_block += 1
            row_count = stop_offset - start_offset
            row_chunk_refs.append(np.full(row_count, chunk_id, dtype=np.int32))
            row_offset_refs.append(np.arange(start_offset, stop_offset, dtype=np.int32))
            block_index = next_block

    if not row_chunk_refs:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return np.concatenate(row_chunk_refs), np.concatenate(row_offset_refs)


class CachedBlockDataset(TorchDataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        chunk_root: Path,
        row_chunk_ids: np.ndarray,
        row_offsets: np.ndarray,
    ) -> None:
        self.chunk_root = chunk_root
        self.row_chunk_ids = row_chunk_ids
        self.row_offsets = row_offsets
        self._chunk_cache: dict[int, Dataset] = {}

    def __len__(self) -> int:
        return int(self.row_offsets.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        chunk_id = int(self.row_chunk_ids[index])
        row_offset = int(self.row_offsets[index])
        if chunk_id < 0 or row_offset < 0:
            raise KeyError(f"cached row {index} is missing from the block cache")
        dataset = self._load_chunk(chunk_id)
        row = dataset[row_offset]
        return {key: row[key] for key in DATA_COLUMNS}

    def _load_chunk(self, chunk_id: int) -> Dataset:
        cached = self._chunk_cache.get(chunk_id)
        if cached is not None:
            return cached
        dataset = load_from_disk(str(self.chunk_root / f"{chunk_id:08d}"))
        self._chunk_cache[chunk_id] = dataset
        return dataset


def prepare_split(
    *,
    split: str,
    dataset_name: str,
    revision: str | None,
    max_samples: int | None,
    seed: int,
    dataset_seed: int | None,
    cache_root: Path,
    startup_callback=None,
) -> PreparedSplit:
    """캐시된 블록에서 sample plan에 따라 PreparedSplit 구성. 캐시 미스 시 FileNotFoundError."""
    source_root, descriptor = ensure_source_root(
        dataset_name=dataset_name,
        revision=revision,
        cache_root=cache_root,
    )
    catalog = ensure_split_catalog(
        source_root=source_root,
        descriptor=descriptor,
        split=split,
        startup_callback=startup_callback,
    )
    sample_plan = plan_sample(catalog, seed, max_samples, split=split)
    resolved_dataset_seed = resolve_dataset_seed(dataset_seed)
    cache_paths = resolve_block_cache_paths(
        source_root,
        split,
        resolved_dataset_seed,
        CANONICAL_SHUFFLE_BUFFER_SIZE,
    )
    cache = load_or_init_block_cache(
        cache_paths,
        dataset_seed=resolved_dataset_seed,
        shuffle_buffer_size=CANONICAL_SHUFFLE_BUFFER_SIZE,
        total_rows=int(catalog["total_rows"]),
    )
    if sample_plan.selected_blocks.size > 0 and not np.all(cache.cached[sample_plan.selected_blocks]):
        raise FileNotFoundError(f"split {split} cache is missing requested blocks")

    execution_runs = compress_execution_runs(
        sample_plan.selected_bitmap,
        cache.cached,
        stop_block=sample_plan.execution_block_count,
    )
    if any(run.kind == "take_remote" for run in execution_runs):
        raise FileNotFoundError(f"split {split} cache is missing requested blocks")
    row_chunk_ids, row_offsets = _build_cached_row_refs(execution_runs, cache)

    dataset = run_startup_stage(
        startup_callback,
        stage="load local cache",
        split=split,
        operation=lambda: CachedBlockDataset(
            chunk_root=cache_paths.chunk_root,
            row_chunk_ids=row_chunk_ids,
            row_offsets=row_offsets,
        ),
        dataset_seed=resolved_dataset_seed,
        requested_rows=sample_plan.requested_rows,
        effective_rows=sample_plan.effective_rows,
        required_blocks=sample_plan.required_blocks,
        candidate_blocks=sample_plan.candidate_blocks,
        planner_mode=sample_plan.planner_mode,
        base_take_prob=sample_plan.base_take_prob,
        run_count=len(execution_runs),
    )
    return PreparedSplit(dataset=dataset)


def _as_shape(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"shape must be a list or tuple, got {type(value)!r}")
    shape = tuple(int(dim) for dim in value)
    if len(shape) != 3:
        raise ValueError(f"expected a 3D tensor shape, got {shape!r}")
    return shape  # type: ignore[return-value]


def _decode_image(buffer: Any, shape: Any, *, dtype: np.dtype[Any], expected_channels: int) -> np.ndarray:
    raw = np.frombuffer(as_bytes(buffer), dtype=dtype)
    resolved_shape = _as_shape(shape)
    expected_size = math.prod(resolved_shape)
    if raw.size != expected_size:
        raise ValueError(f"buffer size mismatch for shape {resolved_shape}: expected {expected_size}, got {raw.size}")

    # CHW/HWC 자동 감지 — 채널 수로 축 판별
    image = raw.reshape(resolved_shape)
    if image.shape[-1] == expected_channels and image.shape[0] != expected_channels:
        chw = np.transpose(image, (2, 0, 1))
    elif image.shape[0] == expected_channels:
        chw = image
    else:
        raise ValueError(f"could not infer channel dimension from shape {image.shape!r}")

    return np.ascontiguousarray(chw, dtype=np.float32)


def decode_row(row: dict[str, Any], *, include_metadata: bool = True) -> dict[str, Any]:
    """원시 바이너리 행을 CHW float32 numpy 배열로 디코딩. ``include_metadata=True`` 시 메타데이터 포함."""
    sar = _decode_image(row["sar"], row["sar_shape"], dtype=np.float32, expected_channels=SAR_CHANNELS)
    cloudy = _decode_image(row["cloudy"], row["opt_shape"], dtype=np.int16, expected_channels=OPTICAL_CHANNELS) / 10000.0
    target = _decode_image(row["target"], row["opt_shape"], dtype=np.int16, expected_channels=OPTICAL_CHANNELS) / 10000.0
    decoded = {"sar": sar, "cloudy": cloudy, "target": target}
    if include_metadata:
        decoded["meta"] = {
            "season": str(row.get("season", "")),
            "scene": str(row.get("scene", "")),
            "patch": str(row.get("patch", "")),
        }
    return decoded


def build_collate_fn(*, include_metadata: bool = True):
    """행 리스트를 디코딩하여 배치 텐서로 묶는 collate 함수 생성."""

    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty batch")

        decoded_rows = [decode_row(row, include_metadata=include_metadata) for row in rows]
        first = decoded_rows[0]
        batch_size = len(decoded_rows)
        sar_batch = torch.empty((batch_size, *first["sar"].shape), dtype=torch.float32)
        cloudy_batch = torch.empty((batch_size, *first["cloudy"].shape), dtype=torch.float32)
        target_batch = torch.empty((batch_size, *first["target"].shape), dtype=torch.float32)

        metadata = {"season": [], "scene": [], "patch": []}
        for index, decoded in enumerate(decoded_rows):
            sar_batch[index].copy_(torch.from_numpy(decoded["sar"]))
            cloudy_batch[index].copy_(torch.from_numpy(decoded["cloudy"]))
            target_batch[index].copy_(torch.from_numpy(decoded["target"]))
            if include_metadata:
                meta = decoded["meta"]
                metadata["season"].append(meta["season"])
                metadata["scene"].append(meta["scene"])
                metadata["patch"].append(meta["patch"])

        batch = {"sar": sar_batch, "cloudy": cloudy_batch, "target": target_batch}
        if include_metadata:
            batch["meta"] = metadata
        return batch

    return collate


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
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    """PreparedSplit에서 DataLoader를 생성. 분산 환경 시 DistributedSampler 자동 적용."""
    dataloader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": build_collate_fn(include_metadata=include_metadata),
        "num_workers": num_workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "drop_last": drop_last if training else False,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    generator = torch.Generator()
    generator.manual_seed(seed + epoch)

    if is_distributed():
        sampler = DistributedSampler(
            prepared.dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=training,
            seed=seed,
        )
        sampler.set_epoch(epoch)
        dataloader_kwargs["sampler"] = sampler
        dataloader_kwargs["shuffle"] = False
    else:
        dataloader_kwargs["shuffle"] = training
        dataloader_kwargs["generator"] = generator if training else None

    return DataLoader(prepared.dataset, **dataloader_kwargs)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """배치 dict 내 텐서를 대상 디바이스로 non-blocking 전송."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


__all__ = [
    "CachedBlockDataset",
    "PreparedSplit",
    "build_collate_fn",
    "build_dataloader",
    "decode_row",
    "move_batch_to_device",
    "prepare_split",
    "resolve_num_workers",
    "seed_everything",
    "seed_worker",
]
