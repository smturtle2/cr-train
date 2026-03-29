from __future__ import annotations

import base64
import csv
import hashlib
import inspect
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal, TypedDict, cast

import numpy as np
import torch
from .runtime import configure_runtime
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader, IterableDataset

configure_runtime()

Stage = Literal["train", "val", "test"]
SplitStrategy = Literal["official", "seeded_scene"]
TensorLayout = Literal["channels_first", "channels_last"]
LoaderProfile = Literal["strict_repro", "throughput"]

SEASON_ORDER = ("spring", "summer", "fall", "winter")
STAGE_ORDER: tuple[Stage, Stage, Stage] = ("train", "val", "test")
SAR_CHANNELS = {2}
OPTICAL_CHANNELS = {13}
DEFAULT_DATASET_NAME = "Hermanni/sen12mscr"
DEFAULT_DATASET_REVISION = "main"


class SampleMetadata(TypedDict):
    season: str
    scene: str
    patch: str
    source_shard: str
    sar_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    sar_dtype: str
    optical_dtype: str


class DecodedSample(TypedDict):
    sar: torch.Tensor
    cloudy: torch.Tensor
    target: torch.Tensor
    metadata: SampleMetadata


Transform = Callable[[DecodedSample], DecodedSample]
SceneSplitResolver = Callable[["DataModuleConfig"], Mapping[Stage, Sequence["SceneShard"]]]
DatasetLoader = Callable[[Sequence[str], Stage], HFIterableDataset]


@dataclass(frozen=True, order=True)
class SceneShard:
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
        return f"hf://datasets/{dataset_name}@{revision}/{self.relative_path}"


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.val, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split ratios must be non-negative")
        if not np.isclose(sum(values), 1.0):
            raise ValueError("split ratios must sum to 1.0")

    def as_mapping(self) -> Mapping[Stage, float]:
        return {"train": self.train, "val": self.val, "test": self.test}


@dataclass(frozen=True)
class ShuffleConfig:
    enabled: bool = True
    buffer_size: int = 16
    reshard_num_shards: int = 1024

    def __post_init__(self) -> None:
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.reshard_num_shards <= 0:
            raise ValueError("reshard_num_shards must be positive")


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    profile: LoaderProfile = "strict_repro"
    in_order: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.profile == "strict_repro" and self.num_workers != 0:
            raise ValueError("strict_repro profile requires num_workers=0")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")


@dataclass(frozen=True)
class DataModuleConfig:
    dataset_name: str = DEFAULT_DATASET_NAME
    revision: str = DEFAULT_DATASET_REVISION
    split_strategy: SplitStrategy = "seeded_scene"
    split_ratios: SplitRatios = field(default_factory=SplitRatios)
    seed: int = 0
    shuffle: ShuffleConfig = field(default_factory=ShuffleConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    tensor_layout: TensorLayout = "channels_first"
    distributed_rank: int = 0
    distributed_world_size: int = 1

    def __post_init__(self) -> None:
        if self.distributed_world_size <= 0:
            raise ValueError("distributed_world_size must be positive")
        if not 0 <= self.distributed_rank < self.distributed_world_size:
            raise ValueError("distributed_rank must be within distributed_world_size")


def _sort_scene_shards(shards: Sequence[SceneShard]) -> list[SceneShard]:
    season_index = {season: index for index, season in enumerate(SEASON_ORDER)}
    return sorted(shards, key=lambda shard: (season_index[shard.season], int(shard.scene)))


@lru_cache(maxsize=1)
def _official_scene_rows() -> tuple[tuple[Stage, SceneShard], ...]:
    resource = files("cr_train.resources").joinpath("official_scene_splits.csv")
    rows: list[tuple[Stage, SceneShard]] = []
    with resource.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stage = cast(Stage, row["stage"])
            rows.append((stage, SceneShard(season=row["season"], scene=row["scene"])))
    return tuple(rows)


@lru_cache(maxsize=1)
def official_scene_splits() -> dict[Stage, tuple[SceneShard, ...]]:
    splits: dict[Stage, list[SceneShard]] = {stage: [] for stage in STAGE_ORDER}
    for stage, shard in _official_scene_rows():
        splits[stage].append(shard)
    return {stage: tuple(_sort_scene_shards(shards)) for stage, shards in splits.items()}


@lru_cache(maxsize=1)
def all_scene_shards() -> tuple[SceneShard, ...]:
    unique = {shard for _, shard in _official_scene_rows()}
    return tuple(_sort_scene_shards(list(unique)))


def _allocate_counts(total: int, ratios: SplitRatios) -> dict[Stage, int]:
    raw = {stage: total * ratio for stage, ratio in ratios.as_mapping().items()}
    counts = {stage: int(np.floor(value)) for stage, value in raw.items()}
    remainder = total - sum(counts.values())
    ranked = sorted(
        STAGE_ORDER,
        key=lambda stage: (raw[stage] - counts[stage], -STAGE_ORDER.index(stage)),
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
    split_ratios: SplitRatios,
    scene_catalog: Sequence[SceneShard] | None = None,
) -> dict[Stage, tuple[SceneShard, ...]]:
    catalog = tuple(scene_catalog) if scene_catalog is not None else all_scene_shards()
    by_season: dict[str, list[SceneShard]] = {season: [] for season in SEASON_ORDER}
    for shard in catalog:
        by_season[shard.season].append(shard)

    assigned: dict[Stage, list[SceneShard]] = {stage: [] for stage in STAGE_ORDER}
    for season in SEASON_ORDER:
        shards = _sort_scene_shards(by_season[season])
        season_rng = random.Random(_season_seed(seed, season))
        season_rng.shuffle(shards)
        counts = _allocate_counts(len(shards), split_ratios)

        cursor = 0
        for stage in STAGE_ORDER:
            next_cursor = cursor + counts[stage]
            assigned[stage].extend(shards[cursor:next_cursor])
            cursor = next_cursor

    return {stage: tuple(_sort_scene_shards(shards)) for stage, shards in assigned.items()}


def resolve_scene_splits(config: DataModuleConfig) -> dict[Stage, tuple[SceneShard, ...]]:
    if config.split_strategy == "official":
        return official_scene_splits()
    return seeded_scene_splits(seed=config.seed, split_ratios=config.split_ratios)


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


def _maybe_relayout(array: np.ndarray, layout: TensorLayout, channel_counts: set[int]) -> np.ndarray:
    if array.ndim != 3:
        return array
    if layout == "channels_first":
        if array.shape[-1] in channel_counts and array.shape[0] not in channel_counts:
            return np.moveaxis(array, -1, 0)
    else:
        if array.shape[0] in channel_counts and array.shape[-1] not in channel_counts:
            return np.moveaxis(array, 0, -1)
    return array


def _decode_tensor(
    blob: bytes | bytearray | memoryview | str,
    *,
    dtype: np.dtype[Any],
    shape: Sequence[int],
    layout: TensorLayout,
    channel_counts: set[int],
) -> torch.Tensor:
    raw = _coerce_bytes(blob)
    array = np.frombuffer(raw, dtype=dtype).reshape(tuple(int(dim) for dim in shape))
    array = _maybe_relayout(array, layout, channel_counts)
    return torch.from_numpy(np.ascontiguousarray(array))


def decode_sample(sample: Mapping[str, Any], tensor_layout: TensorLayout = "channels_first") -> DecodedSample:
    sar_dtype = np.dtype(sample.get("dtype", "float32"))
    optical_dtype = np.dtype("int16")
    sar_shape = tuple(int(dim) for dim in sample["sar_shape"])
    opt_shape = tuple(int(dim) for dim in sample["opt_shape"])
    sar = _decode_tensor(
        sample["sar"],
        dtype=sar_dtype,
        shape=sar_shape,
        layout=tensor_layout,
        channel_counts=SAR_CHANNELS,
    )
    cloudy = _decode_tensor(
        sample["cloudy"],
        dtype=optical_dtype,
        shape=opt_shape,
        layout=tensor_layout,
        channel_counts=OPTICAL_CHANNELS,
    )
    target = _decode_tensor(
        sample["target"],
        dtype=optical_dtype,
        shape=opt_shape,
        layout=tensor_layout,
        channel_counts=OPTICAL_CHANNELS,
    )
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
    return {"sar": sar, "cloudy": cloudy, "target": target, "metadata": metadata}


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _apply_reshard(dataset: Any, target_num_shards: int) -> Any:
    parameters = inspect.signature(dataset.reshard).parameters
    if "num_shards" in parameters:
        return dataset.reshard(num_shards=target_num_shards)
    return dataset.reshard()


def _default_dataset_loader(urls: Sequence[str], _: Stage) -> HFIterableDataset:
    if not urls:
        return cast(HFIterableDataset, _EmptyIterable())
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


class SEN12MSCRStreamingDataset(IterableDataset[DecodedSample]):
    def __init__(
        self,
        source: HFIterableDataset,
        *,
        tensor_layout: TensorLayout = "channels_first",
        transform: Transform | None = None,
        epoch: int = 0,
    ) -> None:
        super().__init__()
        self.source = source
        self.tensor_layout = tensor_layout
        self.transform = transform
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[DecodedSample]:
        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(self.epoch)
        for row in self.source:
            sample = decode_sample(row, tensor_layout=self.tensor_layout)
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample


class SEN12MSCRDataModule:
    def __init__(
        self,
        config: DataModuleConfig,
        *,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        dataset_loader: DatasetLoader | None = None,
        scene_split_resolver: SceneSplitResolver | None = None,
    ) -> None:
        self.config = config
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.dataset_loader = dataset_loader or _default_dataset_loader
        resolved = (
            {
                stage: tuple(_sort_scene_shards(list(shards)))
                for stage, shards in scene_split_resolver(config).items()
            }
            if scene_split_resolver is not None
            else resolve_scene_splits(config)
        )
        self.scene_splits = {stage: tuple(resolved.get(stage, ())) for stage in STAGE_ORDER}

    def scenes_for(self, stage: Stage) -> tuple[SceneShard, ...]:
        return self.scene_splits[stage]

    def _urls_for_stage(self, stage: Stage) -> list[str]:
        return [
            shard.resolve_url(self.config.dataset_name, self.config.revision)
            for shard in self.scenes_for(stage)
        ]

    def _should_shuffle(self, stage: Stage) -> bool:
        return stage == "train" and self.config.shuffle.enabled

    def _stage_transform(self, stage: Stage) -> Transform | None:
        return self.train_transform if stage == "train" else self.eval_transform

    def make_stage_dataset(self, stage: Stage, *, epoch: int = 0) -> SEN12MSCRStreamingDataset:
        urls = self._urls_for_stage(stage)
        dataset = self.dataset_loader(urls, stage)

        if self._should_shuffle(stage):
            dataset = _apply_reshard(dataset, self.config.shuffle.reshard_num_shards)
            dataset = dataset.shuffle(seed=self.config.seed, buffer_size=self.config.shuffle.buffer_size)

        if self.config.distributed_world_size > 1:
            dataset = split_dataset_by_node(
                dataset,
                rank=self.config.distributed_rank,
                world_size=self.config.distributed_world_size,
            )

        wrapped = SEN12MSCRStreamingDataset(
            dataset,
            tensor_layout=self.config.tensor_layout,
            transform=self._stage_transform(stage),
            epoch=epoch,
        )
        return wrapped

    def make_dataloader(
        self,
        stage: Stage,
        *,
        epoch: int = 0,
        collate_fn: Callable[[list[DecodedSample]], Any] | None = None,
    ) -> DataLoader[Any]:
        dataset = self.make_stage_dataset(stage, epoch=epoch)
        generator = torch.Generator()
        generator.manual_seed(self.config.seed + epoch)

        loader_kwargs: dict[str, Any] = {
            "batch_size": self.config.loader.batch_size,
            "num_workers": self.config.loader.num_workers,
            "collate_fn": collate_fn,
            "pin_memory": self.config.loader.pin_memory,
            "drop_last": self.config.loader.drop_last,
            "worker_init_fn": _seed_worker,
            "generator": generator,
            "persistent_workers": self.config.loader.persistent_workers,
            "in_order": self.config.loader.in_order,
        }
        if self.config.loader.num_workers > 0 and self.config.loader.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.config.loader.prefetch_factor

        return DataLoader(dataset, **loader_kwargs)

    def train_dataloader(
        self,
        *,
        epoch: int = 0,
        collate_fn: Callable[[list[DecodedSample]], Any] | None = None,
    ) -> DataLoader[Any]:
        return self.make_dataloader("train", epoch=epoch, collate_fn=collate_fn)

    def val_dataloader(
        self,
        *,
        epoch: int = 0,
        collate_fn: Callable[[list[DecodedSample]], Any] | None = None,
    ) -> DataLoader[Any]:
        return self.make_dataloader("val", epoch=epoch, collate_fn=collate_fn)

    def test_dataloader(
        self,
        *,
        epoch: int = 0,
        collate_fn: Callable[[list[DecodedSample]], Any] | None = None,
    ) -> DataLoader[Any]:
        return self.make_dataloader("test", epoch=epoch, collate_fn=collate_fn)
