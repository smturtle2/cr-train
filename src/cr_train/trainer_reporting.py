from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from .data.constants import WARMUP_TIMELINE_WIDTH
from .data.runtime import _compact_warmup_timeline


PRINTED_STARTUP_STAGES: dict[str, set[str]] = {
    "warm split cache": {"done"},
}


def _resolve_summary_timeline_width(*, split: str) -> int:
    terminal_width = shutil.get_terminal_size(fallback=(WARMUP_TIMELINE_WIDTH + 48, 24)).columns
    reserved_width = len(f" cache {split} | warm | hit=0 miss=0 runs=0") + 10
    return max(12, terminal_width - reserved_width)


def serialize_value(value: Any) -> Any:
    if isinstance(value, (Path, torch.device)):
        return str(value)
    return value


def should_print_startup(record: dict[str, Any]) -> bool:
    status = record.get("status")
    stage = str(record.get("stage", "startup"))
    return status == "error" or status in PRINTED_STARTUP_STAGES.get(stage, set())


def format_cache_summary(event: dict[str, Any]) -> str:
    split = str(event.get("split", "unknown"))
    timeline = _compact_warmup_timeline(
        str(event.get("timeline", "")).strip(),
        max_chars=_resolve_summary_timeline_width(split=split),
    )
    cached_blocks = int(event.get("cached_blocks", 0))
    missing_blocks = int(event.get("missing_blocks", 0))
    run_count = int(event.get("run_count", 0))
    prefix = f"cache {split}" if not timeline else f"{timeline} cache {split}"
    if missing_blocks == 0:
        return f"{prefix} | cache-hit | blocks={cached_blocks} runs={run_count}"
    return f"{prefix} | warm | hit={cached_blocks} miss={missing_blocks} runs={run_count}"


def format_startup_message(event: dict[str, Any]) -> str:
    stage = str(event.get("stage", "startup"))
    split = str(event.get("split", "unknown"))
    if stage == "warm split cache":
        return format_cache_summary(event)

    parts = ["startup", f"split={split}", f"stage={stage}"]
    for field in (
        "max_samples",
        "dataset_seed",
        "requested_rows",
        "effective_rows",
        "required_blocks",
        "candidate_blocks",
        "planner_mode",
        "base_take_prob",
        "cached_blocks",
        "missing_blocks",
        "run_count",
        "resolved_blocks",
        "epoch",
    ):
        if field in event and not (field == "max_samples" and event[field] is None):
            parts.append(f"{field}={event[field]}")
    if "cache_only" in event:
        parts.append(f"cache_only={str(bool(event['cache_only'])).lower()}")
    if event.get("status") == "error" and event.get("error"):
        parts.append(f"error={event['error']}")
    return " | ".join(parts)
