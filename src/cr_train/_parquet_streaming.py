from __future__ import annotations

from copy import deepcopy
from collections.abc import Sequence

import pyarrow as pa
import pyarrow.dataset as pa_ds
from datasets import IterableDataset as HFIterableDataset
from datasets import load_dataset
from datasets.iterable_dataset import ArrowExamplesIterable, _merge_gen_kwargs

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


def default_fragment_scan_options(
    io_profile: IOProfile,
) -> pa_ds.ParquetFragmentScanOptions | None:
    if io_profile != "smooth":
        return None
    options = pa_ds.ParquetFragmentScanOptions()
    options.cache_options = pa.CacheOptions(
        range_size_limit=128 << 20,
        prefetch_limit=1,
        lazy=True,
    )
    return options


def load_streaming_parquet_dataset(
    urls: Sequence[str],
    *,
    io_profile: IOProfile,
    fragment_scan_options: pa_ds.ParquetFragmentScanOptions | None,
) -> HFIterableDataset:
    configure_runtime(io_profile=io_profile)
    return load_dataset(
        "parquet",
        data_files={"train": list(urls)},
        split="train",
        streaming=True,
        columns=list(PARQUET_COLUMNS),
        fragment_scan_options=fragment_scan_options,
    )


def expand_parquet_row_groups(source: HFIterableDataset) -> HFIterableDataset:
    ex_iterable = source._ex_iterable
    expanded_kwargs = list(
        ex_iterable.generate_more_kwargs_fn(
            files=ex_iterable.kwargs["files"],
            row_groups_list=[],
        )
    )
    merged_kwargs = _merge_gen_kwargs(expanded_kwargs)
    expanded_ex_iterable = ArrowExamplesIterable(
        ex_iterable.generate_tables_fn,
        merged_kwargs,
        ex_iterable.generate_more_kwargs_fn,
    )
    return HFIterableDataset(
        ex_iterable=expanded_ex_iterable,
        info=source._info.copy(),
        split=source._split,
        formatting=source._formatting,
        distributed=deepcopy(source._distributed),
        token_per_repo_id=source._token_per_repo_id,
    )
