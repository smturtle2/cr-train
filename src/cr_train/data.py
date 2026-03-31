"""SEN12MS-CR datasets with built-in benchmark-style preprocessing."""

from __future__ import annotations

import csv
import hashlib
import os
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal, TypedDict, cast

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from huggingface_hub import get_token
from torch.utils.data import DataLoader, IterableDataset, default_collate

Stage = Literal["train", "val", "test"]
SplitStrategy = Literal["official", "seeded_scene"]

SEASON_ORDER = ("spring", "summer", "fall", "winter")
STAGE_ORDER: tuple[Stage, Stage, Stage] = ("train", "val", "test")
DEFAULT_DATASET_NAME = "Hermanni/sen12mscr"
DEFAULT_DATASET_REVISION = "e2facda8700dd26cb4cbd5c5d9c82d15f10c38c6"
DEFAULT_SPLIT = "official"
VALID_SPLIT_STRATEGIES: frozenset[SplitStrategy] = frozenset(("official", "seeded_scene"))
DEFAULT_SHUFFLE_BUFFER_SIZE = 8
DEFAULT_SEEDED_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
DEFAULT_AUTO_PREFETCH_FACTOR = 2
MIN_AUTO_TRAIN_WORKERS = 1
AUTO_TRAIN_WORKER_DIVISOR = 4
OPTICAL_MIN = 0.0
OPTICAL_MAX = 10000.0
SAR_DB_MIN = -25.0
SAR_DB_MAX = 0.0
SAR_DTYPE = np.dtype("float32")
OPTICAL_DTYPE = np.dtype("int16")
PARQUET_COLUMNS = (
    "sar",
    "cloudy",
    "target",
    "sar_shape",
    "opt_shape",
    "season",
    "scene",
    "patch",
)


class SampleMetadata(TypedDict):
    """Per-sample metadata preserved through batching for debugging and reproducibility."""

    season: str
    scene: str
    patch: str
    source_shard: str


class SEN12MSCRSample(TypedDict):
    """Standard sample schema consumed by the simplified supervised trainer."""

    inputs: tuple[torch.Tensor, torch.Tensor]
    target: torch.Tensor
    metadata: SampleMetadata


class SampleBatchMetadata(TypedDict):
    """Batch metadata after DataLoader collation."""

    season: list[str]
    scene: list[str]
    patch: list[str]
    source_shard: list[str]


class SEN12MSCRBatch(TypedDict):
    """Standard batched sample schema consumed by the trainer."""

    inputs: tuple[torch.Tensor, torch.Tensor]
    target: torch.Tensor
    metadata: SampleBatchMetadata


DatasetLoader = Callable[[Sequence[str], Stage], Any]
SceneSplitResolver = Callable[[SplitStrategy, int], Mapping[Stage, Sequence["SceneShard"]]]


@dataclass(frozen=True)
class HFTokenStatus:
    configured: bool
    source: Literal["env", "cached", "none"]
    applied_to_hf: bool


def _resolve_hf_token() -> str | None:
    return get_token()


def hf_token_status() -> HFTokenStatus:
    """Return how Hugging Face authentication is configured for dataset access."""

    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return HFTokenStatus(configured=True, source="env", applied_to_hf=True)
    if _resolve_hf_token() is not None:
        return HFTokenStatus(configured=True, source="cached", applied_to_hf=True)
    return HFTokenStatus(configured=False, source="none", applied_to_hf=False)


def hf_token_configured() -> bool:
    """Return whether Hugging Face authentication is configured."""

    return hf_token_status().configured


@dataclass(frozen=True, order=True)
class SceneShard:
    """A single `season/scene` parquet shard in SEN12MS-CR."""

    season: str
    scene: str

    @property
    def relative_path(self) -> str:
        return f"{self.season}/scene_{self.scene}.parquet"

    @property
    def source_id(self) -> str:
        return f"{self.season}:{self.scene}"

    def resolve_url(
        self,
        dataset_name: str = DEFAULT_DATASET_NAME,
        revision: str = DEFAULT_DATASET_REVISION,
    ) -> str:
        # dataset revision을 URL에 박아 재현성을 깨지 않게 한다.
        return f"hf://datasets/{dataset_name}@{revision}/{self.relative_path}"


@dataclass(frozen=True)
class _LoaderOptions:
    batch_size: int
    seed: int = 0
    streaming: bool = True
    split: SplitStrategy = DEFAULT_SPLIT
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE
    num_workers: int | None = None
    pin_memory: bool = False
    timeout: float = 0.0
    prefetch_factor: int | None = None
    persistent_workers: bool | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.split not in VALID_SPLIT_STRATEGIES:
            raise ValueError(
                f"split must be one of {sorted(VALID_SPLIT_STRATEGIES)!r}"
            )
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive when provided")
        if self.num_workers == 0:
            if self.timeout > 0:
                raise ValueError("timeout requires num_workers > 0")
            if self.prefetch_factor is not None:
                raise ValueError("prefetch_factor requires num_workers > 0")
            if self.persistent_workers:
                raise ValueError("persistent_workers requires num_workers > 0")


def _sort_scene_shards(shards: Sequence[SceneShard]) -> list[SceneShard]:
    season_index = {season: index for index, season in enumerate(SEASON_ORDER)}

    def scene_sort_key(shard: SceneShard) -> tuple[int, int]:
        return season_index[shard.season], int(shard.scene)

    return sorted(shards, key=scene_sort_key)


@lru_cache(maxsize=1)
def _official_scene_rows() -> tuple[tuple[Stage, SceneShard], ...]:
    # 번들된 official split CSV는 프로세스당 한 번만 읽는다.
    resource = files("cr_train.resources").joinpath("official_scene_splits.csv")
    rows: list[tuple[Stage, SceneShard]] = []
    with resource.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stage = cast(Stage, row["stage"])
            rows.append((stage, SceneShard(season=row["season"], scene=row["scene"])))
    return tuple(rows)


def _normalize_scene_splits(
    splits: Mapping[Stage, Sequence[SceneShard]],
) -> tuple[tuple[Stage, tuple[SceneShard, ...]], ...]:
    return tuple(
        (stage, tuple(_sort_scene_shards(splits.get(stage, ()))))
        for stage in STAGE_ORDER
    )


@lru_cache(maxsize=1)
def _official_scene_split_items() -> tuple[tuple[Stage, tuple[SceneShard, ...]], ...]:
    splits: dict[Stage, list[SceneShard]] = {stage: [] for stage in STAGE_ORDER}
    for stage, shard in _official_scene_rows():
        splits[stage].append(shard)
    return _normalize_scene_splits(splits)


def official_scene_splits() -> dict[Stage, tuple[SceneShard, ...]]:
    """Return the bundled official scene-level train/val/test split."""

    return dict(_official_scene_split_items())


@lru_cache(maxsize=1)
def _all_scene_shards() -> tuple[SceneShard, ...]:
    unique = {shard for _, shards in _official_scene_split_items() for shard in shards}
    return tuple(_sort_scene_shards(list(unique)))


def _allocate_counts(total: int) -> dict[Stage, int]:
    # season별 shard 수가 작아도 합계가 정확히 맞도록 나머지를 큰 소수점 순으로 배분한다.
    raw = {stage: total * ratio for stage, ratio in DEFAULT_SEEDED_SPLIT_RATIOS.items()}
    counts = {stage: int(np.floor(value)) for stage, value in raw.items()}
    remainder = total - sum(counts.values())

    def remainder_rank(stage: Stage) -> tuple[float, int]:
        return raw[stage] - counts[stage], -STAGE_ORDER.index(stage)

    ranked = sorted(STAGE_ORDER, key=remainder_rank, reverse=True)
    for stage in ranked[:remainder]:
        counts[stage] += 1
    return counts


def _season_seed(seed: int, season: str) -> int:
    digest = hashlib.sha256(f"{seed}:{season}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def seeded_scene_splits(
    *,
    seed: int,
    scene_catalog: Sequence[SceneShard] | None = None,
) -> dict[Stage, tuple[SceneShard, ...]]:
    """Create deterministic custom splits with fixed season-stratified 80/10/10 ratios."""

    catalog = tuple(scene_catalog) if scene_catalog is not None else _all_scene_shards()
    by_season: dict[str, list[SceneShard]] = {season: [] for season in SEASON_ORDER}
    for shard in catalog:
        by_season[shard.season].append(shard)

    assigned: dict[Stage, list[SceneShard]] = {stage: [] for stage in STAGE_ORDER}
    for season in SEASON_ORDER:
        shards = _sort_scene_shards(by_season[season])
        season_rng = random.Random(_season_seed(seed, season))
        season_rng.shuffle(shards)
        counts = _allocate_counts(len(shards))

        cursor = 0
        for stage in STAGE_ORDER:
            next_cursor = cursor + counts[stage]
            assigned[stage].extend(shards[cursor:next_cursor])
            cursor = next_cursor

    return dict(_normalize_scene_splits(assigned))


def _resolve_scene_splits(
    split: SplitStrategy,
    seed: int,
    scene_split_resolver: SceneSplitResolver | None,
) -> dict[Stage, tuple[SceneShard, ...]]:
    if split not in VALID_SPLIT_STRATEGIES:
        raise ValueError(f"split must be one of {sorted(VALID_SPLIT_STRATEGIES)!r}")
    if scene_split_resolver is not None:
        return dict(_normalize_scene_splits(scene_split_resolver(split, seed)))
    if split == "official":
        return official_scene_splits()
    if split == "seeded_scene":
        return seeded_scene_splits(seed=seed)
    raise AssertionError(f"unreachable split strategy: {split}")


def _decode_tensor(blob: bytes, *, dtype: np.dtype[Any], shape: tuple[int, ...]) -> torch.Tensor:
    array = np.frombuffer(blob, dtype=dtype).reshape(shape)
    # 데이터셋은 HWC(256, 256, C) 형태로 저장되어 있으므로 CHW로 변환한다.
    if array.ndim == 3:
        array = np.moveaxis(array, -1, 0)
    return torch.from_numpy(np.array(array, copy=True))


def _preprocess_optical(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor.to(torch.float32), OPTICAL_MIN, OPTICAL_MAX) / OPTICAL_MAX


def _preprocess_sar(tensor: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(tensor.to(torch.float32), SAR_DB_MIN, SAR_DB_MAX)
    return (clipped - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN)


def decode_sample(sample: Mapping[str, Any]) -> SEN12MSCRSample:
    """Decode one raw Hugging Face row into preprocessed CHW tensors and metadata."""

    sar_shape = tuple(int(dim) for dim in sample["sar_shape"])
    opt_shape = tuple(int(dim) for dim in sample["opt_shape"])

    sar = _preprocess_sar(
        _decode_tensor(sample["sar"], dtype=SAR_DTYPE, shape=sar_shape)
    )
    cloudy = _preprocess_optical(
        _decode_tensor(sample["cloudy"], dtype=OPTICAL_DTYPE, shape=opt_shape)
    )
    target = _preprocess_optical(
        _decode_tensor(sample["target"], dtype=OPTICAL_DTYPE, shape=opt_shape)
    )

    scene = str(sample["scene"])
    metadata: SampleMetadata = {
        "season": str(sample["season"]),
        "scene": scene,
        "patch": str(sample["patch"]),
        "source_shard": f"{sample['season']}/scene_{scene}.parquet",
    }
    return {
        "inputs": (sar, cloudy),
        "target": target,
        "metadata": metadata,
    }


def _seed_worker(worker_id: int) -> None:
    _ = worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _dataset_num_shards(dataset: Any) -> int | None:
    source = getattr(dataset, "source", dataset)
    num_shards = getattr(source, "num_shards", None)
    if isinstance(num_shards, int) and num_shards > 0:
        return num_shards
    return None


def _auto_train_worker_budget(dataset: Any) -> int:
    cpu_count = os.cpu_count() or (MIN_AUTO_TRAIN_WORKERS * AUTO_TRAIN_WORKER_DIVISOR)
    budget = max(MIN_AUTO_TRAIN_WORKERS, cpu_count // AUTO_TRAIN_WORKER_DIVISOR)
    dataset_num_shards = _dataset_num_shards(dataset)
    if dataset_num_shards is not None:
        budget = min(budget, dataset_num_shards)
    return budget


def _auto_eval_worker_budget(dataset: Any, *, streaming: bool) -> int:
    if not streaming:
        return 0
    dataset_num_shards = _dataset_num_shards(dataset)
    if dataset_num_shards is not None:
        return min(1, dataset_num_shards)
    return 1


def _resolve_stage_num_workers(stage: Stage, options: _LoaderOptions, dataset: Any) -> int:
    if options.num_workers is not None:
        return options.num_workers
    if stage == "train":
        return _auto_train_worker_budget(dataset)
    return _auto_eval_worker_budget(dataset, streaming=options.streaming)


def _resolve_stage_prefetch_factor(num_workers: int, options: _LoaderOptions) -> int | None:
    if num_workers == 0:
        return None
    if options.prefetch_factor is not None:
        return options.prefetch_factor
    return DEFAULT_AUTO_PREFETCH_FACTOR


def _resolve_stage_persistent_workers(num_workers: int, options: _LoaderOptions) -> bool | None:
    if num_workers == 0:
        return None
    if options.persistent_workers is not None:
        return options.persistent_workers
    return False


def _resolve_stage_timeout(num_workers: int, options: _LoaderOptions) -> float:
    if num_workers == 0:
        return 0.0
    return float(options.timeout)


def _expected_tensor_shape(
    rows: Sequence[Mapping[str, Any]],
    *,
    shape_key: str,
) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in rows[0][shape_key])
    for row in rows[1:]:
        candidate = tuple(int(dim) for dim in row[shape_key])
        if candidate != shape:
            raise ValueError(f"inconsistent {shape_key} values within a batch: {shape!r} != {candidate!r}")
    return shape


def _decode_tensor_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    blob_key: str,
    shape_key: str,
    dtype: np.dtype[Any],
) -> torch.Tensor:
    shape = _expected_tensor_shape(rows, shape_key=shape_key)
    output_shape = (shape[-1], *shape[:-1]) if len(shape) == 3 else shape
    batch = np.empty((len(rows), *output_shape), dtype=dtype)
    for index, row in enumerate(rows):
        array = np.frombuffer(row[blob_key], dtype=dtype).reshape(shape)
        if array.ndim == 3:
            array = np.moveaxis(array, -1, 0)
        batch[index] = array
    return torch.from_numpy(batch)


def _decode_batch(rows: Sequence[Mapping[str, Any]]) -> SEN12MSCRBatch:
    sar = _preprocess_sar(
        _decode_tensor_batch(rows, blob_key="sar", shape_key="sar_shape", dtype=SAR_DTYPE)
    )
    cloudy = _preprocess_optical(
        _decode_tensor_batch(rows, blob_key="cloudy", shape_key="opt_shape", dtype=OPTICAL_DTYPE)
    )
    target = _preprocess_optical(
        _decode_tensor_batch(rows, blob_key="target", shape_key="opt_shape", dtype=OPTICAL_DTYPE)
    )
    metadata: SampleBatchMetadata = {
        "season": [str(row["season"]) for row in rows],
        "scene": [str(row["scene"]) for row in rows],
        "patch": [str(row["patch"]) for row in rows],
        "source_shard": [
            f"{row['season']}/scene_{row['scene']}.parquet"
            for row in rows
        ],
    }
    return {
        "inputs": (sar, cloudy),
        "target": target,
        "metadata": metadata,
    }


def _shuffle_stream_rows(
    rows: Iterator[Any],
    *,
    buffer_size: int,
    seed: str,
) -> Iterator[Any]:
    rng = random.Random(seed)
    buffer: list[Any] = []
    for row in rows:
        if len(buffer) == buffer_size:
            index = rng.randrange(buffer_size)
            yield buffer[index]
            buffer[index] = row
            continue
        buffer.append(row)
    rng.shuffle(buffer)
    yield from buffer


def _collate_sen12mscr_rows(rows: Sequence[Any]) -> Any:
    if not rows:
        raise ValueError("received an empty batch from DataLoader")
    first = rows[0]
    if not isinstance(first, Mapping):
        return default_collate(rows)
    if "inputs" in first and "target" in first:
        return default_collate(rows)
    return _decode_batch(cast(Sequence[Mapping[str, Any]], rows))


class _StreamingSourceAdapter(IterableDataset[Any]):
    """Wrap plain iterables so PyTorch treats streaming custom loaders correctly."""

    def __init__(self, source: Any) -> None:
        super().__init__()
        self.source = source

    @property
    def num_shards(self) -> int | None:
        value = getattr(self.source, "num_shards", None)
        return value if isinstance(value, int) else None

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(epoch)

    def __iter__(self):
        iterator = iter(self.source)
        try:
            yield from iterator
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


class _HFStreamingDataset(IterableDataset[Any]):
    """Recreate the official HF streaming dataset and shuffle train samples locally."""

    def __init__(
        self,
        *,
        urls: Sequence[str],
        stage: Stage,
        seed: int,
        shuffle_buffer_size: int,
    ) -> None:
        super().__init__()
        self._urls = tuple(urls)
        self._stage = stage
        self._seed = seed
        self._shuffle_buffer_size = shuffle_buffer_size
        self._epoch = torch.tensor(0).share_memory_()

    @property
    def num_shards(self) -> int:
        return len(self._urls)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.add_(int(epoch) - int(self._epoch.item()))

    def _build_source(self) -> HFIterableDataset:
        return cast(
            HFIterableDataset,
            _load_parquet_source(
                self._urls,
                streaming=True,
            ),
        )

    def __iter__(self):
        iterator = iter(self._build_source())
        rows: Any = iterator
        if self._stage == "train":
            worker_info = torch.utils.data.get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            rows = _shuffle_stream_rows(
                iterator,
                buffer_size=self._shuffle_buffer_size,
                seed=f"{self._seed}:{int(self._epoch.item())}:{worker_id}",
            )
        try:
            yield from rows
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


def _load_parquet_source(
    urls: Sequence[str],
    *,
    streaming: bool,
) -> HFIterableDataset | HFDataset:
    token = _resolve_hf_token()
    return load_dataset(
        "parquet",
        data_files={"train": list(urls)},
        split="train",
        streaming=streaming,
        columns=list(PARQUET_COLUMNS),
        token=token,
    )


def _stage_urls(stage: Stage, splits: Mapping[Stage, Sequence[SceneShard]]) -> list[str]:
    return [shard.resolve_url() for shard in splits[stage]]


def _build_stage_dataset(
    stage: Stage,
    options: _LoaderOptions,
    *,
    splits: Mapping[Stage, Sequence[SceneShard]],
    dataset_loader: DatasetLoader | None,
) -> Any:
    urls = _stage_urls(stage, splits)
    if options.streaming:
        if dataset_loader is None:
            return _HFStreamingDataset(
                urls=urls,
                stage=stage,
                seed=options.seed,
                shuffle_buffer_size=options.shuffle_buffer_size,
            )
        source = dataset_loader(urls, stage)
        if not isinstance(source, _StreamingSourceAdapter):
            source = _StreamingSourceAdapter(source)
        return source
    return _load_parquet_source(
        urls,
        streaming=False,
    )


def _build_stage_dataloader(
    stage: Stage,
    options: _LoaderOptions,
    *,
    splits: Mapping[Stage, Sequence[SceneShard]],
    dataset_loader: DatasetLoader | None,
) -> DataLoader[Any]:
    dataset = _build_stage_dataset(
        stage,
        options,
        splits=splits,
        dataset_loader=dataset_loader,
    )
    generator = torch.Generator()
    generator.manual_seed(options.seed)
    num_workers = _resolve_stage_num_workers(stage, options, dataset)
    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": options.batch_size,
        "num_workers": num_workers,
        "pin_memory": options.pin_memory,
        "timeout": _resolve_stage_timeout(num_workers, options),
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "collate_fn": _collate_sen12mscr_rows,
    }
    if not options.streaming:
        dataloader_kwargs["shuffle"] = stage == "train"
    resolved_prefetch_factor = _resolve_stage_prefetch_factor(num_workers, options)
    resolved_persistent_workers = _resolve_stage_persistent_workers(num_workers, options)
    if num_workers > 0:
        if resolved_persistent_workers is not None:
            dataloader_kwargs["persistent_workers"] = resolved_persistent_workers
        if resolved_prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = resolved_prefetch_factor
    return DataLoader(**dataloader_kwargs)


def build_sen12mscr_loaders(
    batch_size: int,
    *,
    streaming: bool = True,
    seed: int = 0,
    split: SplitStrategy = DEFAULT_SPLIT,
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
    num_workers: int | None = None,
    pin_memory: bool = False,
    timeout: float = 0.0,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
    _dataset_loader: DatasetLoader | None = None,
    _scene_split_resolver: SceneSplitResolver | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Build `(train_loader, val_loader, test_loader)` for SEN12MS-CR training."""

    options = _LoaderOptions(
        batch_size=batch_size,
        streaming=streaming,
        seed=seed,
        split=split,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        timeout=timeout,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    splits = _resolve_scene_splits(options.split, options.seed, _scene_split_resolver)
    loaders = tuple(
        _build_stage_dataloader(
            cast(Stage, stage),
            options,
            splits=splits,
            dataset_loader=_dataset_loader,
        )
        for stage in STAGE_ORDER
    )
    return cast(tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]], loaders)
