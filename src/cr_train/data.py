"""SEN12MS-CR streaming data pipeline using official HF dataset splits."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import get_token
from torch.utils.data import DataLoader, default_collate

DEFAULT_DATASET = "Hermanni/sen12mscr"
DEFAULT_SHUFFLE_BUFFER_SIZE = 64

OPTICAL_MIN = 0.0
OPTICAL_MAX = 10000.0
SAR_DB_MIN = -25.0
SAR_DB_MAX = 0.0
SAR_DTYPE = np.dtype("float32")
OPTICAL_DTYPE = np.dtype("int16")
PARQUET_COLUMNS = (
    "sar", "cloudy", "target",
    "sar_shape", "opt_shape",
    "season", "scene", "patch",
)

HF_SPLITS: tuple[tuple[str, str], ...] = (
    ("train", "train"),
    ("val", "validation"),
    ("test", "test"),
)


# -- Batch types -------------------------------------------------------------

class SampleMetadata(TypedDict):
    season: str
    scene: str
    patch: str
    source_shard: str


class SampleBatchMetadata(TypedDict):
    season: list[str]
    scene: list[str]
    patch: list[str]
    source_shard: list[str]


class SEN12MSCRBatch(TypedDict):
    inputs: tuple[torch.Tensor, torch.Tensor]
    target: torch.Tensor
    metadata: SampleBatchMetadata


# -- Preprocessing / decode ---------------------------------------------------

@dataclass(frozen=True)
class _TensorField:
    blob_key: str
    shape_key: str
    dtype: np.dtype[Any]
    preprocess: Callable[[torch.Tensor], torch.Tensor]


def _normalize_tensor(
    tensor: torch.Tensor,
    *,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    clipped = torch.clamp(tensor.to(torch.float32), min_value, max_value)
    return (clipped - min_value) / (max_value - min_value)


def _preprocess_optical(tensor: torch.Tensor) -> torch.Tensor:
    return _normalize_tensor(tensor, min_value=OPTICAL_MIN, max_value=OPTICAL_MAX)


def _preprocess_sar(tensor: torch.Tensor) -> torch.Tensor:
    return _normalize_tensor(tensor, min_value=SAR_DB_MIN, max_value=SAR_DB_MAX)


SAR_FIELD = _TensorField(
    blob_key="sar",
    shape_key="sar_shape",
    dtype=SAR_DTYPE,
    preprocess=_preprocess_sar,
)
OPTICAL_INPUT_FIELD = _TensorField(
    blob_key="cloudy",
    shape_key="opt_shape",
    dtype=OPTICAL_DTYPE,
    preprocess=_preprocess_optical,
)
OPTICAL_TARGET_FIELD = _TensorField(
    blob_key="target",
    shape_key="opt_shape",
    dtype=OPTICAL_DTYPE,
    preprocess=_preprocess_optical,
)


def _expected_tensor_shape(
    rows: Sequence[Mapping[str, Any]],
    *,
    shape_key: str,
) -> tuple[int, ...]:
    shape = tuple(int(d) for d in rows[0][shape_key])
    for row in rows[1:]:
        if tuple(int(d) for d in row[shape_key]) != shape:
            raise ValueError(f"inconsistent {shape_key} within a batch")
    return shape


def _decode_tensor_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    blob_key: str,
    shape_key: str,
    dtype: np.dtype[Any],
) -> torch.Tensor:
    shape = _expected_tensor_shape(rows, shape_key=shape_key)
    out_shape = (shape[-1], *shape[:-1]) if len(shape) == 3 else shape
    batch = np.empty((len(rows), *out_shape), dtype=dtype)
    for index, row in enumerate(rows):
        arr = np.frombuffer(row[blob_key], dtype=dtype).reshape(shape)
        if arr.ndim == 3:
            arr = np.moveaxis(arr, -1, 0)
        batch[index] = arr
    return torch.from_numpy(batch)


def _decode_field(rows: Sequence[Mapping[str, Any]], field: _TensorField) -> torch.Tensor:
    decoded = _decode_tensor_batch(
        rows,
        blob_key=field.blob_key,
        shape_key=field.shape_key,
        dtype=field.dtype,
    )
    return field.preprocess(decoded)


def _build_batch_metadata(rows: Sequence[Mapping[str, Any]]) -> SampleBatchMetadata:
    return {
        "season": [str(row["season"]) for row in rows],
        "scene": [str(row["scene"]) for row in rows],
        "patch": [str(row["patch"]) for row in rows],
        "source_shard": [f"{row['season']}/scene_{row['scene']}.parquet" for row in rows],
    }


def _decode_raw_batch(rows: Sequence[Mapping[str, Any]]) -> SEN12MSCRBatch:
    return {
        "inputs": (
            _decode_field(rows, SAR_FIELD),
            _decode_field(rows, OPTICAL_INPUT_FIELD),
        ),
        "target": _decode_field(rows, OPTICAL_TARGET_FIELD),
        "metadata": _build_batch_metadata(rows),
    }


def _decode_batch(rows: Sequence[Mapping[str, Any]]) -> SEN12MSCRBatch:
    return _decode_raw_batch(rows)


def _is_predecoded_batch(first: Mapping[str, Any]) -> bool:
    return "inputs" in first and "target" in first


def _collate_sen12mscr_rows(rows: Sequence[Any]) -> Any:
    if not rows:
        raise ValueError("received an empty batch from DataLoader")
    first = rows[0]
    if not isinstance(first, Mapping):
        return default_collate(rows)
    if _is_predecoded_batch(first):
        return default_collate(rows)
    return _decode_batch(cast(Sequence[Mapping[str, Any]], rows))


# -- Worker seeding -----------------------------------------------------------

def _seed_worker(worker_id: int) -> None:
    _ = worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# -- Public API ---------------------------------------------------------------

def _load_split_dataset(
    dataset: str,
    *,
    split: str,
    token: str | None,
) -> Any:
    return load_dataset(
        dataset,
        split=split,
        streaming=True,
        columns=list(PARQUET_COLUMNS),
        token=token,
    )


def _maybe_shuffle_train_dataset(
    dataset: Any,
    stage: str,
    *,
    seed: int,
    shuffle_buffer_size: int,
) -> Any:
    if stage != "train":
        return dataset
    return dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)


def _build_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_sen12mscr_rows,
    )


def build_loaders(
    batch_size: int,
    *,
    dataset: str = DEFAULT_DATASET,
    seed: int = 0,
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Build ``(train, val, test)`` DataLoaders for SEN12MS-CR streaming."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    token = get_token()
    loaders: list[DataLoader[Any]] = []

    for stage, split in HF_SPLITS:
        split_dataset = _load_split_dataset(dataset, split=split, token=token)
        shuffled_dataset = _maybe_shuffle_train_dataset(
            split_dataset,
            stage,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        loaders.append(
            _build_dataloader(
                shuffled_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        )

    return loaders[0], loaders[1], loaders[2]
