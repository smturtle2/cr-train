from __future__ import annotations

import importlib
from typing import Literal

import fsspec
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import tqdm

from datasets import disable_progress_bars as disable_datasets_progress_bars
from huggingface_hub.utils import disable_progress_bars as disable_hf_progress_bars

IOProfile = Literal["smooth", "conservative"]
VALID_IO_PROFILES: frozenset[IOProfile] = frozenset(("smooth", "conservative"))

_CONFIGURED = False
_CONFIGURED_IO_PROFILE: IOProfile | None = None
_PARQUET_PATCHED = False
_PARQUET_IO_PROFILE: IOProfile | None = None


def _scan_behavior(io_profile: IOProfile) -> tuple[int, int, bool]:
    if io_profile == "smooth":
        return (2, 1, True)
    if io_profile == "conservative":
        return (0, 0, False)
    raise ValueError(f"io_profile must be one of {sorted(VALID_IO_PROFILES)!r}")


def _patch_datasets_parquet_reader(io_profile: IOProfile) -> None:
    global _PARQUET_PATCHED, _PARQUET_IO_PROFILE
    if _PARQUET_PATCHED:
        if _PARQUET_IO_PROFILE != io_profile:
            raise ValueError(
                f"runtime already configured for io_profile={_PARQUET_IO_PROFILE!r}; "
                f"cannot switch to {io_profile!r}"
            )
        return

    # datasets 내부 parquet reader를 직접 패치해 streaming 종료와 읽기 동작을 더 안정적으로 맞춘다.
    parquet_mod = importlib.import_module("datasets.packaged_modules.parquet.parquet")
    parquet_cls = parquet_mod.Parquet
    if getattr(parquet_cls, "_cr_train_use_threads_patch", False):
        configured_profile = getattr(parquet_cls, "_cr_train_io_profile", None)
        if configured_profile != io_profile:
            raise ValueError(
                f"runtime already configured for io_profile={configured_profile!r}; "
                f"cannot switch to {io_profile!r}"
            )
        _PARQUET_PATCHED = True
        _PARQUET_IO_PROFILE = io_profile
        return

    key_cls = parquet_mod.Key
    logger = parquet_mod.logger
    batch_readahead, fragment_readahead, use_threads = _scan_behavior(io_profile)

    def patched_generate_tables(self, files, row_groups_list):
        if self.config.features is not None and self.config.columns is not None:
            feature_names = sorted(field.name for field in self.info.features.arrow_schema)
            if feature_names != sorted(self.config.columns):
                raise ValueError(
                    f"Tried to load parquet data with columns '{self.config.columns}' "
                    f"with mismatching features '{self.info.features}'"
                )

        filter_expr = (
            pq.filters_to_expression(self.config.filters)
            if isinstance(self.config.filters, list)
            else self.config.filters
        )
        parquet_file_format = pa_ds.ParquetFileFormat(
            default_fragment_scan_options=self.config.fragment_scan_options
        )

        for file_idx, (file, row_groups) in enumerate(zip(files, row_groups_list)):
            try:
                with fsspec.open(file, "rb").open() as handle:
                    parquet_fragment = parquet_file_format.make_fragment(handle)
                    if row_groups is not None:
                        parquet_fragment = parquet_fragment.subset(row_group_ids=row_groups)
                    if parquet_fragment.row_groups:
                        batch_size = self.config.batch_size or parquet_fragment.row_groups[0].num_rows
                        batches = parquet_fragment.to_batches(
                            batch_size=batch_size,
                            columns=self.config.columns,
                            filter=filter_expr,
                            # 환경에 따라 io profile을 바꿔 load pause와 안정성의 균형을 조절한다.
                            batch_readahead=batch_readahead,
                            fragment_readahead=fragment_readahead,
                            use_threads=use_threads,
                        )
                        for batch_idx, record_batch in enumerate(batches):
                            table = pa.Table.from_batches([record_batch])
                            yield key_cls(file_idx, batch_idx), self._cast_table(table)
            except (pa.ArrowInvalid, ValueError) as exc:
                if self.config.on_bad_files == "error":
                    logger.error(
                        f"Failed to read file '{file}' with error {type(exc).__name__}: {exc}"
                    )
                    raise
                if self.config.on_bad_files == "warn":
                    logger.warning(f"Skipping bad file '{file}'. {type(exc).__name__}: {exc}")
                else:
                    logger.debug(f"Skipping bad file '{file}'. {type(exc).__name__}: {exc}")

    parquet_cls._generate_tables = patched_generate_tables
    parquet_cls._cr_train_use_threads_patch = True
    parquet_cls._cr_train_io_profile = io_profile
    _PARQUET_PATCHED = True
    _PARQUET_IO_PROFILE = io_profile


def configure_runtime(io_profile: IOProfile = "smooth") -> None:
    global _CONFIGURED, _CONFIGURED_IO_PROFILE
    if io_profile not in VALID_IO_PROFILES:
        raise ValueError(f"io_profile must be one of {sorted(VALID_IO_PROFILES)!r}")
    if _CONFIGURED:
        if _CONFIGURED_IO_PROFILE != io_profile:
            raise ValueError(
                f"runtime already configured for io_profile={_CONFIGURED_IO_PROFILE!r}; "
                f"cannot switch to {io_profile!r}"
            )
        return

    # 외부 progress bar를 끄고 parquet patch를 적용해 trainer 출력만 보이도록 정리한다.
    tqdm.tqdm.monitor_interval = 0
    disable_datasets_progress_bars()
    disable_hf_progress_bars()
    _patch_datasets_parquet_reader(io_profile)
    _CONFIGURED = True
    _CONFIGURED_IO_PROFILE = io_profile
