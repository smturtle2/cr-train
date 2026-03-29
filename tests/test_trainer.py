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


class GradStateMetric:
    name = "grad_state"

    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def __call__(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _ = target
        self.calls.append((torch.is_grad_enabled(), outputs.requires_grad))
        return torch.mean(torch.abs(outputs))


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
    assert val_loader.dataset.epochs == []
    assert test_loader.dataset.epochs == []
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "epoch-0002.pt").exists()


def test_trainer_disables_autograd_during_metric_evaluation() -> None:
    metric = GradStateMetric()
    train_loader = DataLoader(_ToyDataset(_make_rows([0.0, 1.0])), batch_size=2, shuffle=False)
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=nn.MSELoss(),
        metrics=[metric],
        config=TrainerConfig(max_epochs=1, train_max_batches=1, show_progress=False),
        train_loader=train_loader,
    )

    history = list(trainer.step())

    assert len(history) == 1
    assert metric.calls == [(False, True)]


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
    assert val_dataset.epochs == []
    assert test_dataset.epochs == []


def test_trainer_progress_uses_stage_descriptions_and_capped_totals(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []
    printed: list[str] = []

    class _FakeProgress:
        def __init__(self) -> None:
            self.descriptions: list[tuple[str, bool]] = []
            self.updates: list[int] = []

        def update(self, value: int) -> None:
            self.updates.append(value)

        def set_description_str(self, value: str, refresh: bool = True) -> None:
            self.descriptions.append((value, refresh))

        def close(self) -> None:
            return None

    def fake_create_progress(*, desc: str, total: int | None, disable: bool, leave: bool) -> _FakeProgress:
        progress = _FakeProgress()
        created.append(
            {
                "desc": desc,
                "total": total,
                "disable": disable,
                "leave": leave,
                "progress": progress,
            }
        )
        return progress

    monkeypatch.setattr(trainer_mod, "_create_progress", fake_create_progress)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: printed.append(" ".join(map(str, args))))

    train_loader = DataLoader(_ToyDataset(_make_rows([0.0, 1.0, 2.0, 3.0])), batch_size=2, shuffle=False)
    val_loader = DataLoader(_ToyDataset(_make_rows([4.0, 5.0])), batch_size=1, shuffle=False)
    test_loader = DataLoader(_ToyDataset(_make_rows([6.0, 7.0])), batch_size=1, shuffle=False)

    model = nn.Linear(1, 1).to(torch.device("cpu"))
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        criterion=nn.MSELoss(),
        metrics=[MAE()],
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

    assert printed == ["epoch 1/1"]
    assert [{key: value for key, value in item.items() if key != "progress"} for item in created] == [
        {"desc": "train", "total": 2, "disable": False, "leave": True},
        {"desc": "val", "total": 1, "disable": False, "leave": True},
        {"desc": "test", "total": 2, "disable": False, "leave": True},
    ]

    train_progress = created[0]["progress"]
    val_progress = created[1]["progress"]
    test_progress = created[2]["progress"]

    assert isinstance(train_progress, _FakeProgress)
    assert isinstance(val_progress, _FakeProgress)
    assert isinstance(test_progress, _FakeProgress)

    assert train_progress.descriptions == [
        ("train | loading first batch...", True),
        ("train | loss=0.5000 mae=0.5000", False),
        ("train | loss=3.5000 mae=1.5000", False),
    ]
    assert train_progress.updates == [1, 1]
    assert val_progress.descriptions == [
        ("val | loading first batch...", True),
        ("val | loss=16.0000 mae=4.0000", False),
    ]
    assert val_progress.updates == [1]
    assert test_progress.descriptions == [
        ("test | loading first batch...", True),
        ("test | loss=36.0000 mae=6.0000", False),
        ("test | loss=42.5000 mae=6.5000", False),
    ]
    assert test_progress.updates == [1, 1]


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


def test_trainer_rejects_invalid_runtime_step_overrides() -> None:
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=nn.MSELoss(),
        metrics=[],
        config=TrainerConfig(max_epochs=1, show_progress=False),
        train_loader=DataLoader(_ToyDataset(_make_rows([0.0])), batch_size=1, shuffle=False),
        val_loader=DataLoader(_ToyDataset(_make_rows([1.0])), batch_size=1, shuffle=False),
    )

    with pytest.raises(ValueError, match="max_epochs must be positive"):
        list(trainer.step(max_epochs=0))
    with pytest.raises(ValueError, match="train_max_batches must be positive when provided"):
        list(trainer.step(train_max_batches=0))
    with pytest.raises(ValueError, match="val_max_batches must be positive when provided"):
        list(trainer.step(val_max_batches=0))


def test_trainer_rejects_invalid_runtime_test_override() -> None:
    model = nn.Linear(1, 1).to(torch.device("cpu"))
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        criterion=nn.MSELoss(),
        metrics=[],
        config=TrainerConfig(max_epochs=1, show_progress=False),
        train_loader=DataLoader(_ToyDataset(_make_rows([0.0])), batch_size=1, shuffle=False),
        test_loader=DataLoader(_ToyDataset(_make_rows([1.0])), batch_size=1, shuffle=False),
    )

    with pytest.raises(ValueError, match="max_batches must be positive when provided"):
        trainer.test(max_batches=0)
