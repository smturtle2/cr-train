from __future__ import annotations

from pathlib import Path

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


def _step_fn(model: nn.Module, batch: dict[str, torch.Tensor], stage: str) -> StepResult:
    predictions = model(batch["inputs"])
    loss = F.mse_loss(predictions, batch["target"])
    mae = F.l1_loss(predictions, batch["target"])
    return StepResult(loss=loss, metrics={"mae": mae})


def test_trainer_runs_fit_and_test(tmp_path: Path) -> None:
    model = nn.Linear(1, 1)
    trainer = Trainer(
        model=model,
        datamodule=_ToyDataModule(),
        step_fn=_step_fn,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.1),
        checkpoint_dir=tmp_path,
    )

    history = trainer.fit(max_epochs=2, train_max_batches=1, val_max_batches=1)
    metrics = trainer.test(test_max_batches=1)

    assert len(history) == 2
    assert trainer.state.epoch == 2
    assert trainer.state.global_step == 2
    assert "loss" in history[0]["train"]
    assert "loss" in metrics
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "epoch-0002.pt").exists()


def test_trainer_can_restore_checkpoint(tmp_path: Path) -> None:
    first = Trainer(
        model=nn.Linear(1, 1),
        datamodule=_ToyDataModule(),
        step_fn=_step_fn,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.1),
        checkpoint_dir=tmp_path,
    )
    first.fit(max_epochs=1, train_max_batches=1, val_max_batches=1)

    restored = Trainer(
        model=nn.Linear(1, 1),
        datamodule=_ToyDataModule(),
        step_fn=_step_fn,
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.1),
        checkpoint_dir=tmp_path,
    )
    restored.load_checkpoint(tmp_path / "last.pt")

    assert restored.state.epoch == 1
    assert restored.state.global_step == 1
