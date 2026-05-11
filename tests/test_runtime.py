from __future__ import annotations

import torch

from cr_train.data import runtime


def _clear_distributed_env(monkeypatch) -> None:
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CR_TRAIN_DIST_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)


def test_setup_distributed_from_env_returns_cpu_without_torchrun_env(monkeypatch) -> None:
    _clear_distributed_env(monkeypatch)
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: False)

    def fail_init_process_group(*args, **kwargs) -> None:
        raise AssertionError("dist.init_process_group should not be called")

    monkeypatch.setattr(runtime.dist, "init_process_group", fail_init_process_group)

    assert runtime.setup_distributed_from_env() == torch.device("cpu")


def test_setup_distributed_from_env_initializes_torchrun_cuda_rank(monkeypatch) -> None:
    _clear_distributed_env(monkeypatch)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "device_count", lambda: 4)
    selected_devices: list[int] = []
    monkeypatch.setattr(runtime.torch.cuda, "set_device", selected_devices.append)
    monkeypatch.setattr(runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: False)
    backends: list[str] = []
    monkeypatch.setattr(
        runtime.dist,
        "init_process_group",
        lambda *, backend: backends.append(backend),
    )

    assert runtime.setup_distributed_from_env() == torch.device("cuda", 1)
    assert selected_devices == [1]
    assert backends == ["nccl"]


def test_setup_distributed_from_env_is_idempotent_when_already_initialized(monkeypatch) -> None:
    _clear_distributed_env(monkeypatch)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)

    def fail_init_process_group(*args, **kwargs) -> None:
        raise AssertionError("dist.init_process_group should not be called")

    monkeypatch.setattr(runtime.dist, "init_process_group", fail_init_process_group)

    assert runtime.setup_distributed_from_env() == torch.device("cpu")


def test_cleanup_distributed_destroys_initialized_group(monkeypatch) -> None:
    monkeypatch.setattr(runtime.dist, "is_available", lambda: True)
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    destroyed: list[bool] = []
    monkeypatch.setattr(runtime.dist, "destroy_process_group", lambda: destroyed.append(True))

    runtime.cleanup_distributed()

    assert destroyed == [True]
