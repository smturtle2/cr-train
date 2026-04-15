from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .data.constants import WARMUP_TIMELINE_WIDTH

PRINTED_STARTUP_STAGES: dict[str, set[str]] = {
    "warm split cache": {"done"},
    "remote retry": {"retry"},
}

_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_SUMMARY_VALUE_WIDTH = 9
_EPOCH_COUNTER_WIDTH = 3


def _format_epoch_label(epoch: int, epochs: int) -> str:
    return f"Epoch {epoch:>{_EPOCH_COUNTER_WIDTH}}/{epochs:>{_EPOCH_COUNTER_WIDTH}}"


def _epoch_label_width() -> int:
    return len(_format_epoch_label(999, 999))


def _summary_separator() -> str:
    return f" {_DIM}│{_RESET} "


def serialize_value(value: Any) -> Any:
    if isinstance(value, (Path, torch.device)):
        return str(value)
    return value


def should_print_startup(record: dict[str, Any]) -> bool:
    status = record.get("status")
    stage = str(record.get("stage", "startup"))
    return status == "error" or status in PRINTED_STARTUP_STAGES.get(stage, set())


def _compact_timeline(timeline: str, *, max_chars: int = WARMUP_TIMELINE_WIDTH) -> str:
    if len(timeline) <= max_chars:
        return timeline
    if max_chars <= 1:
        return timeline[:max_chars]
    head_chars = max(1, (max_chars - 1) // 2)
    tail_chars = max(1, max_chars - head_chars - 1)
    return f"{timeline[:head_chars]}…{timeline[-tail_chars:]}"


def _format_warmup_timeline(value: Any) -> str | None:
    if value in (None, ""):
        return None
    timeline = str(value).translate(str.maketrans({"█": "■", "░": "□"}))
    return _compact_timeline(timeline)


def format_cache_summary(event: dict[str, Any]) -> str:
    split = str(event.get("split", "unknown"))
    selected_block_count = int(event.get("selected_block_count", 0))
    selected_missing_blocks = int(event.get("selected_missing_blocks", 0))
    resolved_blocks = int(event.get("resolved_blocks", 0))
    prefix = f"cache {split}"
    if selected_missing_blocks == 0:
        parts = [prefix, "cache-hit", f"selected: {selected_block_count}, fill: 0/0"]
    else:
        parts = [
            prefix,
            "warm",
            f"selected: {selected_block_count}, fill: {resolved_blocks}/{selected_missing_blocks}",
        ]
    elapsed_sec = event.get("elapsed_sec")
    if elapsed_sec is not None:
        parts.append(f"{float(elapsed_sec):.1f}s")
    timeline = _format_warmup_timeline(event.get("timeline"))
    if timeline is not None:
        parts.append(timeline)
    return _summary_separator().join(parts)


def format_remote_retry_summary(event: dict[str, Any]) -> str:
    split = event.get("split")
    operation = str(event.get("operation", "remote"))
    attempt = int(event.get("attempt", 0))
    max_attempts = int(event.get("max_attempts", 0))
    delay_sec = float(event.get("delay_sec", 0.0))
    error_type = str(event.get("error_type", "error"))
    parts = [
        f"retry {split}" if split is not None else "retry",
        operation,
        f"attempt {attempt}/{max_attempts}",
        f"backoff {delay_sec:.1f}s",
        error_type,
    ]
    if "cache_key" in event:
        parts.append(f"cache_key={event['cache_key']}")
    if "recovery" in event:
        parts.append(f"recovery={event['recovery']}")
    return _summary_separator().join(parts)


def format_startup_message(event: dict[str, Any]) -> str:
    stage = str(event.get("stage", "startup"))
    split = str(event.get("split", "unknown"))
    if stage == "warm split cache":
        return format_cache_summary(event)
    if stage == "remote retry":
        return format_remote_retry_summary(event)

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
        "operation",
        "attempt",
        "max_attempts",
        "delay_sec",
        "error_type",
    ):
        if field in event and not (field == "max_samples" and event[field] is None):
            parts.append(f"{field}={event[field]}")
    if "cache_only" in event:
        parts.append(f"cache_only={str(bool(event['cache_only'])).lower()}")
    if event.get("status") == "error" and event.get("error"):
        parts.append(f"error={event['error']}")
    return _summary_separator().join(parts)


def format_metric_value(value: float) -> str:
    return f"{float(value):.4f}"


def format_learning_rate(value: float) -> str:
    if value == 0.0:
        return "0"

    mantissa, exponent = f"{float(value):.4e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def format_learning_rates(values: Any) -> str | None:
    if not isinstance(values, list) or not values:
        return None
    formatted = [format_learning_rate(float(value)) for value in values]
    if len(formatted) == 1:
        return formatted[0]
    return "[" + ", ".join(formatted) + "]"


def _visible_width(text: str) -> int:
    return len(_ANSI_ESCAPE_RE.sub("", text))


def _pad_visible(text: str, width: int) -> str:
    return text + (" " * max(0, width - _visible_width(text)))


def _left_pad_visible(text: str, width: int) -> str:
    return (" " * max(0, width - _visible_width(text))) + text


def _format_elapsed(elapsed_sec: float | None) -> str | None:
    if elapsed_sec is None:
        return None
    return f"{_DIM}{float(elapsed_sec):.1f}s{_RESET}"


def _format_train_trailer(train: Mapping[str, Any], elapsed_sec: float | None) -> str | None:
    if elapsed_sec is not None:
        return _format_elapsed(elapsed_sec)
    if "samples_per_sec" in train:
        speed = float(train["samples_per_sec"])
        return f"{_DIM}{speed:.1f} samples/s{_RESET}"
    return None


def _summary_field_width(name: str) -> int:
    return len(name) + 1 + _SUMMARY_VALUE_WIDTH


def _format_value_field(name: str, value: str) -> str:
    return f"{name} {_left_pad_visible(value, _SUMMARY_VALUE_WIDTH)}"


def _format_metric_field(name: str, value: float) -> str:
    return _format_value_field(name, format_metric_value(value))


def _format_learning_rate_field(learning_rates: str) -> str:
    return _format_value_field("lr", learning_rates)


def _format_blank_field(name: str) -> str:
    return " " * _summary_field_width(name)


def _format_summary_row(
    *,
    epoch_label: str,
    epoch_width: int,
    split_label: str,
    split_width: int,
    loss: float,
    metrics: Mapping[str, Any],
    show_learning_rate_field: bool = False,
    learning_rates: str | None = None,
    trailer: str | None = None,
) -> str:
    parts = [
        _pad_visible(epoch_label, epoch_width),
        _pad_visible(split_label, split_width),
        _format_metric_field("loss", loss),
    ]
    for name, value in metrics.items():
        parts.append(_format_metric_field(name, float(value)))
    if show_learning_rate_field:
        if learning_rates is not None:
            parts.append(_format_learning_rate_field(learning_rates))
        elif trailer is not None:
            parts.append(_format_blank_field("lr"))
    if trailer is not None:
        parts.append(trailer)
    return _summary_separator().join(parts)


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
    accum_steps: int,
    epochs: int,
    seed: int,
    device: torch.device,
    num_workers: int,
    multiprocessing_context: str | None,
    scheduler_name: str | None,
    scheduler_timing: str,
    scheduler_monitor: str | None,
    grad_clip_norm: float | None,
) -> str:
    header = f"{_BOLD}cr-train{_RESET} {_DIM}── {dataset_name} ── {device}{_RESET}"
    splits = (
        f"  {_DIM}splits{_RESET}  "
        f"train {_BOLD}{_samples_label(max_train_samples)}{_RESET}  "
        f"val {_BOLD}{_samples_label(max_val_samples)}{_RESET}  "
        f"test {_BOLD}{_samples_label(max_test_samples)}{_RESET}"
    )
    config_parts = [
        f"batch {batch_size}",
        f"accum {accum_steps}",
        f"epochs {epochs}",
        f"seed {seed}",
        f"workers {num_workers}",
    ]
    if multiprocessing_context is not None:
        config_parts.append(f"mp {multiprocessing_context}")
    if scheduler_name is not None:
        config_parts.append(f"scheduler {scheduler_name}")
        config_parts.append(f"timing {scheduler_timing}")
    if scheduler_monitor is not None:
        config_parts.append(f"monitor {scheduler_monitor}")
    if grad_clip_norm is not None:
        config_parts.append(f"clip {format_metric_value(grad_clip_norm)}")
    config = f"  {_DIM}config{_RESET}  " + _summary_separator().join(config_parts)
    return f"{header}\n{splits}\n{config}"


def format_train_epoch_row(
    *,
    epoch: int,
    epochs: int,
    train: Mapping[str, Any],
    elapsed_sec: float | None = None,
) -> str:
    learning_rates = format_learning_rates(train.get("lr"))
    return _format_summary_row(
        epoch_label=f"{_BOLD}{_format_epoch_label(epoch, epochs)}{_RESET}",
        epoch_width=_epoch_label_width(),
        split_label=f"{_GREEN}train{_RESET}",
        split_width=len("train"),
        loss=float(train["loss"]),
        metrics=train.get("metrics", {}),
        show_learning_rate_field=True,
        learning_rates=learning_rates,
        trailer=_format_train_trailer(train, elapsed_sec),
    )


def format_val_epoch_row(
    *,
    epochs: int,
    val: Mapping[str, Any],
    train_learning_rates: Any = None,
    elapsed_sec: float | None = None,
) -> str:
    return _format_summary_row(
        epoch_label="",
        epoch_width=_epoch_label_width(),
        split_label=f"{_CYAN}val{_RESET}",
        split_width=len("train"),
        loss=float(val["loss"]),
        metrics=val.get("metrics", {}),
        show_learning_rate_field=train_learning_rates is not None,
        trailer=_format_elapsed(elapsed_sec),
    )


def format_epoch_summary(result: dict[str, Any], *, epochs: int) -> str:
    epoch = result["epoch"]
    train = result["train"]
    train_line = format_train_epoch_row(
        epoch=epoch,
        epochs=epochs,
        train=train,
        elapsed_sec=result.get("train_elapsed_sec", result.get("elapsed_sec")),
    )
    val_line = format_val_epoch_row(
        epochs=epochs,
        val=result["val"],
        train_learning_rates=train.get("lr"),
        elapsed_sec=result.get("val_elapsed_sec"),
    )
    return f"{train_line}\n{val_line}"


def format_test_summary(
    result: dict[str, Any],
    *,
    learning_rates: Any = None,
    elapsed_sec: float | None = None,
) -> str:
    return _format_summary_row(
        epoch_label="",
        epoch_width=_epoch_label_width(),
        split_label=f"{_BOLD}Test{_RESET}",
        split_width=len("train"),
        loss=float(result.get("loss", 0.0)),
        metrics=result.get("metrics", {}),
        show_learning_rate_field=learning_rates is not None,
        trailer=_format_elapsed(elapsed_sec),
    )
