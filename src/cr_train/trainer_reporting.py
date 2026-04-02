from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

PRINTED_STARTUP_STAGES: dict[str, set[str]] = {
    "warm split cache": {"done"},
}

_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"


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
    selected_block_count = int(event.get("selected_block_count", 0))
    selected_missing_blocks = int(event.get("selected_missing_blocks", 0))
    resolved_blocks = int(event.get("resolved_blocks", 0))
    prefix = f"cache {split}"
    elapsed_sec = event.get("elapsed_sec")
    if selected_missing_blocks == 0:
        summary = f"{prefix} | cache-hit | selected: {selected_block_count}, fill: 0/0"
    else:
        summary = f"{prefix} | warm | selected: {selected_block_count}, fill: {resolved_blocks}/{selected_missing_blocks}"
    if elapsed_sec is None:
        return summary
    return f"{summary} | {float(elapsed_sec):.1f}s"


def format_startup_message(event: dict[str, Any]) -> str:
    stage = str(event.get("stage", "startup"))
    split = str(event.get("split", "unknown"))
    if stage == "warm split cache":
        return format_cache_summary(event)

    parts = ["startup", f"split={split}", f"stage={stage}"]
    for field in (
        "max_samples",
        "requested_rows",
        "effective_rows",
        "required_blocks",
        "planner_mode",
        "selected_block_count",
        "cached_selected_blocks",
        "selected_missing_blocks",
        "execution_block_count",
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


def _fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.0001:
        return f"{value:.2e}"
    if abs(value) < 10:
        return f"{value:.4f}"
    return f"{value:.2f}"


def _samples_label(n: int | None) -> str:
    if n is None:
        return "full"
    return f"{n:,}"


def format_config_banner(
    *,
    dataset_name: str,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> str:
    sep = f"{_DIM}│{_RESET}"
    header = f"{_BOLD}cr-train{_RESET} {_DIM}── {dataset_name} ── {device}{_RESET}"
    splits = (
        f"  {_DIM}splits{_RESET}  "
        f"train {_BOLD}{_samples_label(max_train_samples)}{_RESET}  "
        f"val {_BOLD}{_samples_label(max_val_samples)}{_RESET}  "
        f"test {_BOLD}{_samples_label(max_test_samples)}{_RESET}"
    )
    config_parts = [f"batch {batch_size}", f"epochs {epochs}", f"seed {seed}"]
    config = f"  {_DIM}config{_RESET}  " + f"  {sep}  ".join(config_parts)
    return f"{header}\n{splits}\n{config}"


def format_epoch_summary(result: dict[str, Any], *, epochs: int) -> str:
    epoch = result["epoch"]
    train = result["train"]
    val = result["val"]
    sep = f" {_DIM}│{_RESET} "

    parts = [f"{_BOLD}Epoch {epoch}/{epochs}{_RESET}"]

    train_parts = [f"{_GREEN}train{_RESET} loss {_fmt(train['loss'])}"]
    for name, value in train.get("metrics", {}).items():
        train_parts.append(f"{name} {_fmt(value)}")
    parts.append(" ".join(train_parts))

    val_parts = [f"{_CYAN}val{_RESET} loss {_fmt(val['loss'])}"]
    for name, value in val.get("metrics", {}).items():
        val_parts.append(f"{name} {_fmt(value)}")
    parts.append(" ".join(val_parts))

    if "samples_per_sec" in train:
        speed = train["samples_per_sec"]
        parts.append(f"{_DIM}{speed:.1f} samples/s{_RESET}")

    return sep.join(parts)


def format_test_summary(result: dict[str, Any]) -> str:
    sep = f" {_DIM}│{_RESET} "
    parts = [f"{_BOLD}Test{_RESET}"]

    metric_parts = [f"loss {_fmt(result.get('loss', 0.0))}"]
    for name, value in result.get("metrics", {}).items():
        metric_parts.append(f"{name} {_fmt(value)}")
    parts.append(" ".join(metric_parts))

    parts.append(f"{_DIM}{result.get('num_samples', 0):,} samples{_RESET}")
    return sep.join(parts)
