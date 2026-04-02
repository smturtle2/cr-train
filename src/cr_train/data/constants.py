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
CACHE_LAYOUT_VERSION = 10
LOCK_POLL_INTERVAL_SECONDS = 0.1
LOCK_TIMEOUT_SECONDS = 600.0
DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co"
CATALOG_METADATA_WORKERS = 4
BLOCK_SIZE = 64
STOP_BIAS_ALPHA = 8.0
STOP_BIAS_MIX = 0.95
CHUNK_TARGET_BLOCKS = 8
WARMUP_SPEED_EMA_ALPHA = 0.25
WARMUP_DOWNLOAD_SPEED_WINDOW_SEC = 0.5
WARMUP_TIMELINE_WIDTH = 32
StartupCallback = Callable[[Mapping[str, Any]], None]
