from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cr_train import StepResult, Trainer


class _ToyDataModule:
    def __init__(self) -> None:
        self.train_rows = [
            {"inputs": torch.tensor([[0.0]], dtype=torch.float32), "target": torch.tensor([[0.0]])},
            {"inputs": torch.tensor([[1.0]], dtype=torch.float32), "target": torch.tensor([[1.0]])},
            {"inputs": torch.tensor([[2.0]], dtype=torch.float32), "target": torch.tensor([[2.0]])},
            {"inputs": torch.tensor([[3.0]], dtype=torch.float32), "target": torch.tensor([[3.0]])},
        ]
        self.eval_rows = [
            {"inputs": torch.tensor([[4.0]], dtype=torch.float32), "target": torch.tensor([[4.0]])},
            {"inputs": torch.tensor([[5.0]], dtype=torch.float32), "target": torch.tensor([[5.0]])},
        ]

    def train_dataloader(self, *, epoch: int = 0):
        return DataLoader(self.train_rows, batch_size=2, shuffle=False)

    def val_dataloader(self, *, epoch: int = 0):
        return DataLoader(self.eval_rows, batch_size=1, shuffle=False)

    def test_dataloader(self, *, epoch: int = 0):
        return DataLoader(self.eval_rows, batch_size=1, shuffle=False)


class _ToyScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.steps = state["steps"]


def _step_fn(model: nn.Module, batch: dict[str, torch.Tensor], stage: str) -> StepResult:
    predictions = model(batch["inputs"])
    loss = F.mse_loss(predictions, batch["target"])
    mae = F.l1_loss(predictions, batch["target"])
    return StepResult(loss=loss, metrics={"mae": mae})


def test_trainer_runs_fit_and_test(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        datamodule=_ToyDataModule(),
        step_fn=_step_fn,
        checkpoint_dir=tmp_path,
    )

    history = trainer.fit(max_epochs=2, train_max_batches=1, val_max_batches=1)
    metrics = trainer.test(test_max_batches=1)

    assert len(history) == 2
    assert trainer.device == device
    assert trainer.state.epoch == 2
    assert trainer.state.global_step == 2
    assert trainer.optimizer is optimizer
    assert "loss" in history[0]["train"]
    assert "loss" in metrics
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "epoch-0002.pt").exists()


def test_trainer_uses_model_device_and_scheduler_contract(tmp_path: Path) -> None:
    observed_batch_devices: list[torch.device] = []
    observed_scheduler_metrics: list[dict[str, float]] = []
    scheduler = _ToyScheduler()
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def step_fn(model: nn.Module, batch: dict[str, torch.Tensor], stage: str) -> StepResult:
        observed_batch_devices.append(batch["inputs"].device)
        predictions = model(batch["inputs"])
        loss = F.mse_loss(predictions, batch["target"])
        mae = F.l1_loss(predictions, batch["target"])
        return StepResult(loss=loss, metrics={"mae": mae})

    def scheduler_step_fn(current_scheduler: _ToyScheduler, metrics: dict[str, float]) -> None:
        current_scheduler.steps += 1
        observed_scheduler_metrics.append(metrics)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        datamodule=_ToyDataModule(),
        step_fn=step_fn,
        scheduler=scheduler,
        scheduler_step_fn=scheduler_step_fn,
        checkpoint_dir=tmp_path,
    )
    trainer.fit(max_epochs=1, train_max_batches=1, val_max_batches=1)

    restored_model = nn.Linear(1, 1).to(torch.device("cpu"))
    restored_scheduler = _ToyScheduler()
    restored = Trainer(
        model=restored_model,
        optimizer=torch.optim.SGD(restored_model.parameters(), lr=0.1),
        datamodule=_ToyDataModule(),
        step_fn=_step_fn,
        scheduler=restored_scheduler,
        scheduler_step_fn=lambda scheduler, metrics: None,
        checkpoint_dir=tmp_path,
    )
    restored.load_checkpoint(tmp_path / "last.pt")

    assert observed_batch_devices == [torch.device("cpu"), torch.device("cpu")]
    assert scheduler.steps == 1
    assert restored_scheduler.steps == 1
    assert restored.state.epoch == 1
    assert len(observed_scheduler_metrics) == 1
    assert set(observed_scheduler_metrics[0]) == {
        "train/loss",
        "train/mae",
        "val/loss",
        "val/mae",
    }


def test_trainer_requires_paired_scheduler_arguments() -> None:
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError):
        Trainer(
            model=model,
            optimizer=optimizer,
            datamodule=_ToyDataModule(),
            step_fn=_step_fn,
            scheduler=_ToyScheduler(),
        )


def test_trainer_rejects_optimizer_from_other_model() -> None:
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    other_model = nn.Linear(1, 1).to(torch.device("cpu"))

    with pytest.raises(ValueError):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(other_model.parameters(), lr=0.1),
            datamodule=_ToyDataModule(),
            step_fn=_step_fn,
        )
