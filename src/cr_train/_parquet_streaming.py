from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.dataset as pa_ds
from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset

from .runtime import IOProfile, configure_runtime

PARQUET_COLUMNS: tuple[str, ...] = (
    "sar",
    "cloudy",
    "target",
    "sar_shape",
    "opt_shape",
    "season",
    "scene",
    "patch",
)


def default_cache_options(io_profile: IOProfile) -> pa.CacheOptions | None:
    if io_profile != "smooth":
        return None
    return pa.CacheOptions(
        range_size_limit=128 << 20,
        prefetch_limit=1,
        lazy=True,
    )


def _fragment_scan_options(
    cache_options: pa.CacheOptions | None,
) -> pa_ds.ParquetFragmentScanOptions | None:
    if cache_options is None:
        return None
    options = pa_ds.ParquetFragmentScanOptions()
    options.cache_options = cache_options
    return options


def load_streaming_parquet_dataset(
    urls: Sequence[str],
    *,
    io_profile: IOProfile,
    cache_options: pa.CacheOptions | None,
) -> HFIterableDataset:
    configure_runtime(io_profile=io_profile)
    return load_dataset(
        "parquet",
        data_files={"train": list(urls)},
        split="train",
        streaming=True,
        columns=list(PARQUET_COLUMNS),
        fragment_scan_options=_fragment_scan_options(cache_options),
    )


def load_parquet_dataset(
    urls: Sequence[str],
    *,
    io_profile: IOProfile,
) -> HFDataset:
    configure_runtime(io_profile=io_profile)
    return load_dataset(
        "parquet",
        data_files={"train": list(urls)},
        split="train",
        streaming=False,
        columns=list(PARQUET_COLUMNS),
    )
