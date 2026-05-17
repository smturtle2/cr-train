from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


DATASET_ID = "Hermanni/sen12mscr-v2"
SAR_CHANNELS = 2
OPTICAL_CHANNELS = 13
CRPACK_LAYOUT_VERSION = 15
CRPACK_BLOCK_SIZE = 32
LOCK_POLL_INTERVAL_SECONDS = 0.1
LOCK_TIMEOUT_SECONDS = 600.0
BLOCK_SIZE = 64
WARMUP_TIMELINE_WIDTH = 32
StartupCallback = Callable[[Mapping[str, Any]], None]
