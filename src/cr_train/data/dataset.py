from __future__ import annotations

import hashlib
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset, get_worker_info

from .constants import OPTICAL_CHANNELS, SAR_CHANNELS
from .planning import plan_sample
from .runtime import get_rank, get_world_size
from .source import ensure_source_root, ensure_split_catalog, run_startup_stage
from .store import BlockCachePaths, as_bytes, load_block, load_completed_block_index, resolve_block_cache_paths

_TRAIN_ACTIVE_BLOCK_COUNT = 8


@dataclass(slots=True)
class PreparedSplit:
    """DataLoader-ready split backed by the block cache."""

    dataset: TorchIterableDataset[dict[str, Any]]
    num_examples: int


@dataclass(slots=True)
class PreparedSplitState:
    """Static split selection resolved against the local block cache."""

    split: str
    cache_paths: BlockCachePaths
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
                _normalize_sar_tensor(sar_image)
                sar_batch[i].copy_(_apply_spatial_transform(sar_image, params))

                cloudy_image = torch.empty(opt_chw, dtype=torch.float32)
                _decode_image_into(
                    cloudy_image,
                    row["cloudy"],
                    row["opt_shape"],
                    src_dtype=torch.int16,
                    expected_channels=OPTICAL_CHANNELS,
                )
                _normalize_optical_tensor(cloudy_image)
                cloudy_batch[i].copy_(_apply_spatial_transform(cloudy_image, params))

                target_image = torch.empty(opt_chw, dtype=torch.float32)
                _decode_image_into(
                    target_image,
                    row["target"],
                    row["opt_shape"],
                    src_dtype=torch.int16,
                    expected_channels=OPTICAL_CHANNELS,
                )
                _normalize_optical_tensor(target_image)
                target_batch[i].copy_(_apply_spatial_transform(target_image, params))
            else:
                _decode_image_into(
                    sar_batch[i],
                    row["sar"],
                    row["sar_shape"],
                    src_dtype=torch.float32,
                    expected_channels=SAR_CHANNELS,
                )
                _normalize_sar_tensor(sar_batch[i])
                _decode_image_into(
                    cloudy_batch[i],
                    row["cloudy"],
                    row["opt_shape"],
                    src_dtype=torch.int16,
                    expected_channels=OPTICAL_CHANNELS,
                )
                _normalize_optical_tensor(cloudy_batch[i])
                _decode_image_into(
                    target_batch[i],
                    row["target"],
                    row["opt_shape"],
                    src_dtype=torch.int16,
                    expected_channels=OPTICAL_CHANNELS,
                )
                _normalize_optical_tensor(target_batch[i])
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
    return min(4, max(1, cpu_count // 4))


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


def _slice_blocks_for_rank(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    world_size = get_world_size()
    if world_size <= 1:
        return list(blocks)
    rank = get_rank()
    return [block for i, block in enumerate(blocks) if i % world_size == rank]


def _count_rows_from_state(state: PreparedSplitState, blocks: list[dict[str, Any]]) -> int:
    total = 0
    for block in blocks:
        total += int(state.row_counts_by_key[str(block["cache_key"])])
    return total


class CachedBlockIterableDataset(TorchIterableDataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        cache_paths,
        blocks: tuple[dict[str, Any], ...],
        seed: int,
        epoch: int,
        split: str,
        training: bool,
    ) -> None:
        self.cache_paths = cache_paths
        self.blocks = blocks
        self.seed = seed
        self.epoch = epoch
        self.split = split
        self.training = training

    def _load_active_block(self, *, block: dict[str, Any]) -> _ActiveBlockCursor:
        cache_key = str(block["cache_key"])
        rows = load_block(self.cache_paths, cache_key)
        indices = list(range(len(rows)))
        if len(indices) > 1:
            row_rng = random.Random(
                _derive_block_seed(
                    self.seed,
                    split=self.split,
                    epoch=self.epoch,
                    cache_key=cache_key,
                )
            )
            row_rng.shuffle(indices)
        return _ActiveBlockCursor(rows=rows, indices=indices)

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

        def refill_active_blocks() -> None:
            nonlocal next_block_index
            while (
                len(active_blocks) < _TRAIN_ACTIVE_BLOCK_COUNT
                and next_block_index < len(blocks)
            ):
                active_blocks.append(
                    self._load_active_block(block=blocks[next_block_index])
                )
                next_block_index += 1

        refill_active_blocks()
        while active_blocks:
            if not round_robin:
                round_robin = list(active_blocks)
                mix_rng.shuffle(round_robin)

            current_block = round_robin.pop()
            yield current_block.pop()

            if current_block.exhausted:
                active_blocks.remove(current_block)
                round_robin = [
                    candidate for candidate in round_robin if candidate is not current_block
                ]
                refill_active_blocks()

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1
        blocks = [block for index, block in enumerate(self.blocks) if index % worker_count == worker_id]

        if self.training:
            yield from self._iter_training_rows(blocks=blocks, worker_id=worker_id)
            return

        for block in blocks:
            cache_key = str(block["cache_key"])
            rows = load_block(self.cache_paths, cache_key)
            yield from rows


CachedRowDataset = CachedBlockIterableDataset


def _resolve_selected_blocks(
    catalog: dict[str, Any],
    *,
    selected_indices: list[int],
) -> list[dict[str, Any]]:
    catalog_blocks = list(catalog.get("blocks", []))
    return [catalog_blocks[index] for index in selected_indices]


def _resolve_selected_block_row_counts(
    cache_paths: BlockCachePaths,
    selected_blocks: list[dict[str, Any]],
) -> dict[str, int]:
    completed_by_key = load_completed_block_index(cache_paths)
    missing_cache_keys: list[str] = []
    row_counts_by_key: dict[str, int] = {}
    for block in selected_blocks:
        cache_key = str(block["cache_key"])
        row_count = completed_by_key.get(cache_key)
        if row_count is None:
            missing_cache_keys.append(cache_key)
            continue
        row_counts_by_key[cache_key] = row_count
    if missing_cache_keys:
        raise FileNotFoundError(f"split cache is missing requested blocks: {', '.join(missing_cache_keys)}")
    return row_counts_by_key


def resolve_prepared_split_state(
    *,
    split: str,
    dataset_name: str,
    revision: str | None,
    max_samples: int | None,
    seed: int,
    cache_root: Path,
    startup_callback=None,
) -> PreparedSplitState:
    """Resolve the static block selection for a split from the local cache."""
    source_root, descriptor = ensure_source_root(
        dataset_name=dataset_name,
        revision=revision,
        cache_root=cache_root,
        startup_callback=startup_callback,
    )
    catalog = ensure_split_catalog(
        source_root=source_root,
        descriptor=descriptor,
        split=split,
        startup_callback=startup_callback,
    )
    sample_plan = plan_sample(
        catalog,
        seed,
        max_samples,
        split=split,
    )
    cache_paths = resolve_block_cache_paths(source_root, split)
    selected_blocks = _resolve_selected_blocks(
        catalog,
        selected_indices=[int(index) for index in sample_plan.selected_blocks.tolist()],
    )
    row_counts_by_key = _resolve_selected_block_row_counts(cache_paths, selected_blocks)
    return PreparedSplitState(
        split=split,
        cache_paths=cache_paths,
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
    rank_blocks = _slice_blocks_for_rank(ordered_blocks)
    num_examples = _count_rows_from_state(state, rank_blocks)
    dataset = run_startup_stage(
        startup_callback,
        stage="load local cache",
        split=state.split,
        operation=lambda: CachedBlockIterableDataset(
            cache_paths=state.cache_paths,
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
    cache_root: Path,
    startup_callback=None,
) -> PreparedSplit:
    """Build a PreparedSplit from the block cache. Missing blocks are fatal."""
    state = resolve_prepared_split_state(
        split=split,
        dataset_name=dataset_name,
        revision=revision,
        max_samples=max_samples,
        seed=seed,
        cache_root=cache_root,
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
    """Decode one cached row into CHW float32 arrays."""
    sar = _decode_image(row["sar"], row["sar_shape"], dtype=np.float32, expected_channels=SAR_CHANNELS)
    _normalize_sar_numpy(sar)
    cloudy = _decode_image(row["cloudy"], row["opt_shape"], dtype=np.int16, expected_channels=OPTICAL_CHANNELS)
    _normalize_optical_numpy(cloudy)
    target = _decode_image(row["target"], row["opt_shape"], dtype=np.int16, expected_channels=OPTICAL_CHANNELS)
    _normalize_optical_numpy(target)
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
    """Create the split DataLoader for the cached iterable dataset."""
    del seed, epoch
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
    "CachedRowDataset",
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
