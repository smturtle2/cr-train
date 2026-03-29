"""Streaming SEN12MS-CR datasets with built-in benchmark-style preprocessing."""

from __future__ import annotations

import base64
import csv
import hashlib
import multiprocessing as mp
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal, TypedDict, cast

import numpy as np
import torch
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from .runtime import configure_runtime

Stage = Literal["train", "val", "test"]
SplitStrategy = Literal["official", "seeded_scene"]

SEASON_ORDER = ("spring", "summer", "fall", "winter")
STAGE_ORDER: tuple[Stage, Stage, Stage] = ("train", "val", "test")
SAR_CHANNELS = {2}
OPTICAL_CHANNELS = {13}
DEFAULT_DATASET_NAME = "Hermanni/sen12mscr"
DEFAULT_DATASET_REVISION = "e2facda8700dd26cb4cbd5c5d9c82d15f10c38c6"
DEFAULT_SPLIT = "official"
VALID_SPLIT_STRATEGIES: frozenset[SplitStrategy] = frozenset(("official", "seeded_scene"))
DEFAULT_SHUFFLE_BUFFER_SIZE = 16
DEFAULT_SEEDED_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
OPTICAL_MIN = 0.0
OPTICAL_MAX = 10000.0
SAR_DB_MIN = -25.0
SAR_DB_MAX = 0.0


class SampleMetadata(TypedDict):
    """Per-sample metadata preserved through batching for debugging and reproducibility."""

    season: str
    scene: str
    patch: str
    source_shard: str
    sar_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    sar_dtype: str
    optical_dtype: str


class SEN12MSCRSample(TypedDict):
    """Standard sample schema consumed by the simplified supervised trainer."""

    inputs: dict[str, torch.Tensor]
    target: torch.Tensor
    metadata: SampleMetadata


DatasetLoader = Callable[[Sequence[str], Stage], HFIterableDataset]
SceneSplitResolver = Callable[[SplitStrategy, int], Mapping[Stage, Sequence["SceneShard"]]]


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
    split: SplitStrategy = DEFAULT_SPLIT
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE
    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.split not in VALID_SPLIT_STRATEGIES:
            raise ValueError(
                f"split must be one of {sorted(VALID_SPLIT_STRATEGIES)!r}"
            )
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive when provided")
        # worker 프로세스가 없으면 prefetch/persistent worker 옵션도 의미가 없다.
        if self.num_workers == 0:
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

    ranked = sorted(
        STAGE_ORDER,
        key=remainder_rank,
        reverse=True,
    )
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
    # season별 비율을 유지한 채 seed만 바꿔도 항상 같은 split이 나오게 한다.
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


def _coerce_bytes(blob: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, bytearray):
        return bytes(blob)
    if isinstance(blob, memoryview):
        return blob.tobytes()
    if isinstance(blob, str):
        return base64.b64decode(blob.strip('"'))
    raise TypeError(f"unsupported binary payload type: {type(blob)!r}")


def _maybe_channels_first(array: np.ndarray, channel_counts: set[int]) -> np.ndarray:
    if array.ndim != 3:
        return array
    if array.shape[-1] in channel_counts and array.shape[0] not in channel_counts:
        return np.moveaxis(array, -1, 0)
    return array


def _decode_tensor(
    blob: bytes | bytearray | memoryview | str,
    *,
    dtype: np.dtype[Any],
    shape: Sequence[int],
    channel_counts: set[int],
) -> torch.Tensor:
    raw = _coerce_bytes(blob)
    array = np.frombuffer(raw, dtype=dtype).reshape(tuple(int(dim) for dim in shape))
    array = _maybe_channels_first(array, channel_counts)
    # parquet/buffer 메모리와 분리된 torch tensor를 만들어 이후 변형이 안전하게 되게 한다.
    return torch.from_numpy(np.array(array, copy=True))


def _preprocess_optical(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor.to(torch.float32), OPTICAL_MIN, OPTICAL_MAX) / OPTICAL_MAX


def _preprocess_sar(tensor: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(tensor.to(torch.float32), SAR_DB_MIN, SAR_DB_MAX)
    return (clipped - SAR_DB_MIN) / (SAR_DB_MAX - SAR_DB_MIN)


def _decode_optical_sample(
    sample: Mapping[str, Any],
    field: str,
    *,
    dtype: np.dtype[Any],
    shape: Sequence[int],
) -> torch.Tensor:
    return _preprocess_optical(
        _decode_tensor(
            sample[field],
            dtype=dtype,
            shape=shape,
            channel_counts=OPTICAL_CHANNELS,
        )
    )


def decode_sample(sample: Mapping[str, Any]) -> SEN12MSCRSample:
    """Decode one raw Hugging Face row into preprocessed CHW tensors and metadata."""

    sar_dtype = np.dtype(sample.get("dtype", "float32"))
    optical_dtype = np.dtype("int16")
    sar_shape = tuple(int(dim) for dim in sample["sar_shape"])
    opt_shape = tuple(int(dim) for dim in sample["opt_shape"])

    # raw payload 세 개를 모두 CHW float tensor로 바꾸고 범위를 정규화한다.
    sar = _preprocess_sar(
        _decode_tensor(
            sample["sar"],
            dtype=sar_dtype,
            shape=sar_shape,
            channel_counts=SAR_CHANNELS,
        )
    )
    cloudy = _decode_optical_sample(sample, "cloudy", dtype=optical_dtype, shape=opt_shape)
    target = _decode_optical_sample(sample, "target", dtype=optical_dtype, shape=opt_shape)

    scene = str(sample["scene"])
    metadata: SampleMetadata = {
        "season": str(sample["season"]),
        "scene": scene,
        "patch": str(sample["patch"]),
        "source_shard": f"{sample['season']}/scene_{scene}.parquet",
        "sar_shape": sar_shape,
        "opt_shape": opt_shape,
        "sar_dtype": sar_dtype.name,
        "optical_dtype": optical_dtype.name,
    }
    return {
        "inputs": {"sar": sar, "cloudy": cloudy},
        "target": target,
        "metadata": metadata,
    }


def _seed_worker(worker_id: int) -> None:
    _ = worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _apply_reshard(dataset: Any, target_num_shards: int) -> Any:
    try:
        return dataset.reshard(num_shards=target_num_shards)
    except TypeError:
        return dataset.reshard()


def _default_dataset_loader(urls: Sequence[str], _: Stage) -> HFIterableDataset:
    if not urls:
        return cast(HFIterableDataset, _EmptyIterable())
    # runtime patch는 실제 HF streaming loader를 열 때만 적용한다.
    configure_runtime()
    return load_dataset(
        "parquet",
        data_files={"train": list(urls)},
        split="train",
        streaming=True,
    )


class _EmptyIterable:
    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(())

    def reshard(self, num_shards: int) -> "_EmptyIterable":
        _ = num_shards
        return self

    def shuffle(self, *, seed: int, buffer_size: int) -> "_EmptyIterable":
        _ = (seed, buffer_size)
        return self

    def set_epoch(self, epoch: int) -> None:
        _ = epoch


class SEN12MSCRStreamingDataset(IterableDataset[SEN12MSCRSample]):
    """Iterable dataset that decodes and preprocesses SEN12MS-CR rows on the fly."""

    def __init__(self, source: HFIterableDataset, *, epoch: int = 0) -> None:
        super().__init__()
        self.source = source
        # persistent worker에서도 최신 epoch가 보이도록 shared value로 유지한다.
        self._shared_epoch = mp.Value("q", epoch)

    @property
    def epoch(self) -> int:
        with self._shared_epoch.get_lock():
            return int(self._shared_epoch.value)

    def set_epoch(self, epoch: int) -> None:
        """Forward the current epoch to the underlying streaming source."""

        with self._shared_epoch.get_lock():
            self._shared_epoch.value = epoch

    def __iter__(self) -> Iterator[SEN12MSCRSample]:
        current_epoch = self.epoch
        if hasattr(self.source, "set_epoch"):
            # HF iterable dataset은 iteration 시작 시점의 epoch 값으로 shard/example 순서를 결정한다.
            self.source.set_epoch(current_epoch)
        for row in self.source:
            yield decode_sample(row)


def _stage_urls(stage: Stage, splits: Mapping[Stage, Sequence[SceneShard]]) -> list[str]:
    return [shard.resolve_url() for shard in splits[stage]]


def _build_stage_dataset(
    stage: Stage,
    options: _LoaderOptions,
    *,
    splits: Mapping[Stage, Sequence[SceneShard]],
    dataset_loader: DatasetLoader | None,
) -> SEN12MSCRStreamingDataset:
    urls = _stage_urls(stage, splits)
    source = (dataset_loader or _default_dataset_loader)(urls, stage)
    if stage == "train":
        # train만 re-shard 후 shuffle하고, val/test는 고정 순서를 유지한다.
        source = _apply_reshard(source, target_num_shards=1024)
        source = source.shuffle(seed=options.seed, buffer_size=options.shuffle_buffer_size)
    return SEN12MSCRStreamingDataset(source)


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
    # torch worker seed도 dataloader마다 고정해 streaming 순서를 재현 가능하게 맞춘다.
    generator = torch.Generator()
    generator.manual_seed(options.seed)
    dataloader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": options.batch_size,
        "num_workers": options.num_workers,
        "pin_memory": options.pin_memory,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if options.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = options.persistent_workers
        if options.prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = options.prefetch_factor
    return DataLoader(
        **dataloader_kwargs,
    )


def build_sen12mscr_loaders(
    batch_size: int,
    *,
    seed: int = 0,
    split: SplitStrategy = DEFAULT_SPLIT,
    shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    _dataset_loader: DatasetLoader | None = None,
    _scene_split_resolver: SceneSplitResolver | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Build `(train_loader, val_loader, test_loader)` for SEN12MS-CR streaming training.

    The underscored keyword arguments are reserved for tests and internal integration hooks.
    """

    options = _LoaderOptions(
        batch_size=batch_size,
        seed=seed,
        split=split,
        shuffle_buffer_size=shuffle_buffer_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
