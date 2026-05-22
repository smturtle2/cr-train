from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

import torch
import torch.distributed as dist

from .constants import StartupCallback


def emit_startup_event(startup_callback: StartupCallback | None, **event: Any) -> None:
    if startup_callback is not None:
        startup_callback(event)


def run_startup_stage(
    startup_callback: StartupCallback | None,
    *,
    stage: str,
    split: str,
    operation,
    **fields: Any,
):
    emit_startup_event(startup_callback, stage=stage, split=split, status="start", **fields)
    started_at = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        emit_startup_event(
            startup_callback,
            stage=stage,
            split=split,
            status="error",
            elapsed_sec=time.perf_counter() - started_at,
            error=str(exc),
            **fields,
        )
        raise
    emit_startup_event(
        startup_callback,
        stage=stage,
        split=split,
        status="done",
        elapsed_sec=time.perf_counter() - started_at,
        **fields,
    )
    return result


def _render_warmup_timeline(selected_bitmap, *, stop_block: int) -> str:
    if stop_block <= 0:
        return ""
    return "".join("█" if selected_bitmap[i] else "░" for i in range(stop_block))


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _has_distributed_env() -> bool:
    return _env_int("RANK") is not None and _env_int("WORLD_SIZE") is not None


def _shutdown_from_signal(signum: int, _frame: Any) -> None:
    signal.signal(signum, signal.SIG_DFL)
    raise SystemExit(128 + signum)


def install_shutdown_signal_handlers() -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    current = signal.getsignal(signal.SIGTERM)
    if current in (signal.SIG_IGN, _shutdown_from_signal):
        return
    if current != signal.SIG_DFL:
        return
    signal.signal(signal.SIGTERM, _shutdown_from_signal)


def _resolve_local_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = _env_int("LOCAL_RANK")
    if local_rank is None:
        local_rank = 0

    device_count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside the visible CUDA device range "
            f"0..{device_count - 1}"
        )

    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def setup_distributed_from_env() -> torch.device:
    """Initialize torch.distributed from torchrun env vars and return the local device."""
    install_shutdown_signal_handlers()
    device = _resolve_local_cuda_device()
    if not _has_distributed_env():
        return device

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    if not dist.is_initialized():
        backend = os.environ.get("CR_TRAIN_DIST_BACKEND")
        if backend is None:
            backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    return device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_primary() -> bool:
    return get_rank() == 0


__all__ = [
    "_render_warmup_timeline",
    "cleanup_distributed",
    "emit_startup_event",
    "get_rank",
    "get_world_size",
    "install_shutdown_signal_handlers",
    "is_distributed",
    "is_primary",
    "run_startup_stage",
    "setup_distributed_from_env",
]
