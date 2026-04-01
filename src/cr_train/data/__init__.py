from __future__ import annotations

from .constants import (
    BLOCK_SIZE,
    CANONICAL_SHUFFLE_BUFFER_SIZE,
    DATA_COLUMNS,
    DATASET_ID,
    OPTICAL_CHANNELS,
    SAR_CHANNELS,
)
from .dataset import (
    CachedBlockDataset,
    PreparedSplit,
    build_collate_fn,
    build_dataloader,
    decode_row,
    move_batch_to_device,
    prepare_split,
    resolve_num_workers,
    seed_everything,
    seed_worker,
)
from .planning import (
    CachePlan,
    ExecutionRun,
    SamplePlan,
    build_cache_plan,
    compute_base_take_probability,
    compute_take_probability,
    compress_execution_runs,
    compress_frontier_runs,
    plan_sample,
)
from .runtime import WarmupProgressState, ensure_split_cache, get_rank, get_world_size, is_distributed, is_primary
from .source import (
    emit_startup_event,
    ensure_source_root,
    ensure_split_catalog,
    normalize_parquet_uri,
    pq,
    request_json,
    run_startup_stage,
)
from .store import (
    BlockCachePaths,
    SplitBlockCache,
    SplitBlockCacheState,
    file_lock,
    freeze_row,
    read_json,
    remove_tree,
    resolve_cache_root,
    save_dataset_without_progress,
    suppress_hf_datasets_progress_bars,
    write_json_atomic,
)

_compress_execution_runs = compress_execution_runs
_compress_frontier_runs = compress_frontier_runs
_compute_base_take_probability = compute_base_take_probability
_compute_take_probability = compute_take_probability
_emit_startup_event = emit_startup_event
_ensure_source_root = ensure_source_root
_ensure_split_catalog = ensure_split_catalog
_file_lock = file_lock
_freeze_row = freeze_row
_normalize_parquet_uri = normalize_parquet_uri
_plan_sample = plan_sample
_read_json = read_json
_remove_tree = remove_tree
_request_json = request_json
_run_startup_stage = run_startup_stage
_save_dataset_without_progress = save_dataset_without_progress
_suppress_hf_datasets_progress_bars = suppress_hf_datasets_progress_bars
_write_json_atomic = write_json_atomic

__all__ = [
    "BLOCK_SIZE",
    "BlockCachePaths",
    "CANONICAL_SHUFFLE_BUFFER_SIZE",
    "CachePlan",
    "CachedBlockDataset",
    "DATASET_ID",
    "DATA_COLUMNS",
    "ExecutionRun",
    "OPTICAL_CHANNELS",
    "PreparedSplit",
    "SAR_CHANNELS",
    "SamplePlan",
    "SplitBlockCache",
    "SplitBlockCacheState",
    "WarmupProgressState",
    "_compress_execution_runs",
    "_compress_frontier_runs",
    "_compute_base_take_probability",
    "_compute_take_probability",
    "_emit_startup_event",
    "_ensure_source_root",
    "_ensure_split_catalog",
    "_file_lock",
    "_freeze_row",
    "_normalize_parquet_uri",
    "_plan_sample",
    "_read_json",
    "_remove_tree",
    "_request_json",
    "_run_startup_stage",
    "_save_dataset_without_progress",
    "_suppress_hf_datasets_progress_bars",
    "_write_json_atomic",
    "build_cache_plan",
    "build_collate_fn",
    "build_dataloader",
    "compute_base_take_probability",
    "compute_take_probability",
    "decode_row",
    "ensure_split_cache",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_primary",
    "move_batch_to_device",
    "pq",
    "prepare_split",
    "resolve_cache_root",
    "resolve_num_workers",
    "run_startup_stage",
    "seed_everything",
    "seed_worker",
]
