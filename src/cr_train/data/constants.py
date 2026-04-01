from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


DATASET_ID = "Hermanni/sen12mscr"
DATA_COLUMNS = [
    "sar",
    "cloudy",
    "target",
    "sar_shape",
    "opt_shape",
    "season",
    "scene",
    "patch",
]
SAR_CHANNELS = 2
OPTICAL_CHANNELS = 13
CACHE_LAYOUT_VERSION = 7
LOCK_POLL_INTERVAL_SECONDS = 0.1
LOCK_TIMEOUT_SECONDS = 600.0
DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co"
CATALOG_METADATA_WORKERS = 4
DEFAULT_DATASET_SEED = 0
BLOCK_SIZE = 16
CANONICAL_SHUFFLE_BUFFER_SIZE = 1024
CANDIDATE_WINDOW_FACTOR = 4
WARMUP_SPEED_EMA_ALPHA = 0.25
StartupCallback = Callable[[Mapping[str, Any]], None]
