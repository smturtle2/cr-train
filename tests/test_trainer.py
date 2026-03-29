from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

import cr_train.trainer as trainer_mod
from cr_train import MAE, Trainer, TrainerConfig


class _ToyDataset(Dataset[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.epochs: list[int] = []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _ToyStream(IterableDataset[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.epochs: list[int] = []

    def __iter__(self):
        yield from self.rows

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class SquaredError:
    def __call__(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((outputs - target) ** 2)


class DuplicateMAE:
    name = "mae"

    def __call__(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(outputs - target))


def _make_rows(values: list[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value in values:
        rows.append(
            {
                "inputs": torch.tensor([[value]], dtype=torch.float32),
                "target": torch.tensor([[value]], dtype=torch.float32),
                "metadata": {"index": value},
            }
        )
    return rows


def test_trainer_step_yields_epoch_history_and_test_metrics(tmp_path: Path) -> None:
    train_loader = DataLoader(_ToyDataset(_make_rows([0.0, 1.0, 2.0, 3.0])), batch_size=2, shuffle=False)
    val_loader = DataLoader(_ToyDataset(_make_rows([4.0, 5.0])), batch_size=1, shuffle=False)
    test_loader = DataLoader(_ToyDataset(_make_rows([6.0, 7.0])), batch_size=1, shuffle=False)

    model = nn.Linear(1, 1).to(torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        metrics=[MAE(), SquaredError()],
        config=TrainerConfig(
            max_epochs=2,
            train_max_batches=1,
            val_max_batches=1,
            test_max_batches=1,
            checkpoint_dir=tmp_path,
            show_progress=False,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    history = list(trainer.step())
    metrics = trainer.test()

    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2
    assert "loss" in history[0]["train"]
    assert "mae" in history[0]["train"]
    assert "squared_error" in history[0]["train"]
    assert "loss" in history[0]["val"]
    assert "loss" in metrics
    assert "mae" in metrics
    assert "squared_error" in metrics
    assert trainer.state.epoch == 2
    assert trainer.state.global_step == 2
    assert train_loader.dataset.epochs == [0, 1]
    assert val_loader.dataset.epochs == [0, 1]
    assert test_loader.dataset.epochs == [2]
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "epoch-0002.pt").exists()


def test_trainer_progress_defaults_on_and_supports_unsized_loaders() -> None:
    train_dataset = _ToyStream(_make_rows([0.0, 1.0, 2.0]))
    val_dataset = _ToyStream(_make_rows([3.0, 4.0]))
    test_dataset = _ToyStream(_make_rows([5.0, 6.0]))
    model = nn.Linear(1, 1).to(torch.device("cpu"))

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=nn.MSELoss(),
        metrics=[MAE()],
        config=TrainerConfig(max_epochs=1),
        train_loader=DataLoader(train_dataset, batch_size=1),
        val_loader=DataLoader(val_dataset, batch_size=1),
        test_loader=DataLoader(test_dataset, batch_size=1),
    )

    assert trainer.show_progress is True

    history = list(trainer.step())
    metrics = trainer.test()

    assert len(history) == 1
    assert "loss" in history[0]["train"]
    assert "loss" in history[0]["val"]
    assert "loss" in metrics
    assert train_dataset.epochs == [0]
    assert val_dataset.epochs == [0]
    assert test_dataset.epochs == [1]


def test_trainer_progress_uses_stage_descriptions_and_capped_totals(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class _FakeProgress:
        def update(self, value: int) -> None:
            _ = value

        def set_postfix(self, value: object) -> None:
            _ = value

        def close(self) -> None:
            return None

    def fake_create_progress(*, desc: str, total: int | None, disable: bool, leave: bool) -> _FakeProgress:
        created.append(
            {
                "desc": desc,
                "total": total,
                "disable": disable,
                "leave": leave,
            }
        )
        return _FakeProgress()

    monkeypatch.setattr(trainer_mod, "_create_progress", fake_create_progress)

    train_loader = DataLoader(_ToyDataset(_make_rows([0.0, 1.0, 2.0, 3.0])), batch_size=2, shuffle=False)
    val_loader = DataLoader(_ToyDataset(_make_rows([4.0, 5.0])), batch_size=1, shuffle=False)
    test_loader = DataLoader(_ToyDataset(_make_rows([6.0, 7.0])), batch_size=1, shuffle=False)

    model = nn.Linear(1, 1).to(torch.device("cpu"))
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=nn.MSELoss(),
        metrics=[],
        config=TrainerConfig(
            max_epochs=1,
            train_max_batches=10,
            val_max_batches=1,
            test_max_batches=3,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    list(trainer.step())
    trainer.test()

    assert created == [
        {"desc": "epoch 1 train", "total": 2, "disable": False, "leave": False},
        {"desc": "epoch 1 val", "total": 1, "disable": False, "leave": False},
        {"desc": "test", "total": 2, "disable": False, "leave": False},
    ]


def test_trainer_can_restore_checkpoint(tmp_path: Path) -> None:
    train_loader = DataLoader(_ToyDataset(_make_rows([0.0, 1.0])), batch_size=2, shuffle=False)
    val_loader = DataLoader(_ToyDataset(_make_rows([2.0])), batch_size=1, shuffle=False)

    first_model = nn.Linear(1, 1).to(torch.device("cpu"))
    first = Trainer(
        model=first_model,
        optimizer=torch.optim.SGD(first_model.parameters(), lr=0.1),
        criterion=nn.MSELoss(),
        metrics=[MAE()],
        config=TrainerConfig(
            max_epochs=1,
            train_max_batches=1,
            val_max_batches=1,
            checkpoint_dir=tmp_path,
            show_progress=False,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
    )
    list(first.step())

    restored_model = nn.Linear(1, 1).to(torch.device("cpu"))
    restored = Trainer(
        model=restored_model,
        optimizer=torch.optim.SGD(restored_model.parameters(), lr=0.1),
        criterion=nn.MSELoss(),
        metrics=[MAE()],
        config=TrainerConfig(max_epochs=2, checkpoint_dir=tmp_path, show_progress=False),
        train_loader=train_loader,
        val_loader=val_loader,
    )
    restored.load_checkpoint(tmp_path / "last.pt")

    assert restored.state.epoch == 1
    assert restored.state.global_step == 1


def test_trainer_rejects_optimizer_from_other_model() -> None:
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    other_model = nn.Linear(1, 1).to(torch.device("cpu"))

    with pytest.raises(ValueError):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(other_model.parameters(), lr=0.1),
            criterion=nn.MSELoss(),
            metrics=[],
            config=TrainerConfig(max_epochs=1, show_progress=False),
            train_loader=DataLoader(_ToyDataset(_make_rows([0.0])), batch_size=1, shuffle=False),
        )


def test_trainer_rejects_duplicate_metric_names() -> None:
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    with pytest.raises(ValueError):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            criterion=nn.MSELoss(),
            metrics=[MAE(), DuplicateMAE()],
            config=TrainerConfig(max_epochs=1, show_progress=False),
            train_loader=DataLoader(_ToyDataset(_make_rows([0.0])), batch_size=1, shuffle=False),
        )


def test_trainer_rejects_legacy_metric_mapping() -> None:
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    with pytest.raises(TypeError):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            criterion=nn.MSELoss(),
            metrics={"mae": MAE()},
            config=TrainerConfig(max_epochs=1, show_progress=False),
            train_loader=DataLoader(_ToyDataset(_make_rows([0.0])), batch_size=1, shuffle=False),
        )


def test_trainer_rejects_lambda_metrics() -> None:
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    with pytest.raises(ValueError):
        Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            criterion=nn.MSELoss(),
            metrics=[lambda outputs, target: torch.mean(torch.abs(outputs - target))],
            config=TrainerConfig(max_epochs=1, show_progress=False),
            train_loader=DataLoader(_ToyDataset(_make_rows([0.0])), batch_size=1, shuffle=False),
        )


def test_trainer_requires_test_loader_for_test_call() -> None:
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=nn.MSELoss(),
        metrics=[],
        config=TrainerConfig(max_epochs=1, show_progress=False),
        train_loader=DataLoader(_ToyDataset(_make_rows([0.0])), batch_size=1, shuffle=False),
    )

    with pytest.raises(ValueError):
        trainer.test()
