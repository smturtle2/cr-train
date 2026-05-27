from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch.distributed as dist


def test_trainer_runs_real_two_process_ddp_with_global_metric_reductions(tmp_path: Path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    script_path = tmp_path / "ddp_smoke.py"
    output_dir = tmp_path / "run"
    script_path.write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import json
            import os
            from pathlib import Path

            import torch
            import torch.distributed as dist
            from torch import nn
            from torch.utils.data import DataLoader, Dataset

            from cr_train import Trainer, cleanup_distributed, setup_distributed_from_env
            import cr_train.trainer as trainer_mod

            torch.cuda.is_available = lambda: False


            class FakeTqdm:
                instances = []
                writes = []

                def __init__(self, *args, **kwargs) -> None:
                    self.desc = kwargs.get("desc")
                    self.disable = kwargs.get("disable", False)
                    self.postfixes = []
                    self.updates = []
                    FakeTqdm.instances.append(self)

                @staticmethod
                def write(message, *args, **kwargs) -> None:
                    del args, kwargs
                    FakeTqdm.writes.append(str(message))

                def update(self, value: int) -> None:
                    self.updates.append(value)

                def set_postfix_str(self, text: str) -> None:
                    self.postfixes.append(text)

                def close(self) -> None:
                    return None


            class ToyDataset(Dataset):
                def __init__(self, *, rank: int, split: str) -> None:
                    self.rank = rank
                    self.split_offset = {"train": 0, "validation": 10, "test": 20}[split]

                def __len__(self) -> int:
                    return 4

                def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                    value = float(self.split_offset + self.rank * 4 + index)
                    return {
                        "sar": torch.tensor([value], dtype=torch.float32),
                        "cloudy": torch.tensor([1.0], dtype=torch.float32),
                        "target": torch.tensor([value + 1.0], dtype=torch.float32),
                    }


            class ToyModel(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.scale = nn.Parameter(torch.tensor(0.5))

                def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
                    return (sar + cloudy) * self.scale


            def loss_fn(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
                return torch.mean((prediction - batch["target"]) ** 2)


            def mae(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
                return torch.mean(torch.abs(prediction - batch["target"]))


            def fake_build_loader(self, *, split, max_samples, training, epoch_index):
                del max_samples, training, epoch_index
                dataset = ToyDataset(rank=dist.get_rank(), split=split)
                total_batches = 2
                return DataLoader(dataset, batch_size=2, num_workers=0), total_batches


            trainer_mod.Trainer._ensure_training_startup_data = lambda self: None
            trainer_mod.Trainer._build_loader = fake_build_loader
            trainer_mod.tqdm = FakeTqdm

            rank = int(os.environ["RANK"])
            out_dir = Path(os.environ["CR_TRAIN_TEST_OUT"])
            out_dir.mkdir(parents=True, exist_ok=True)
            trainer = None
            try:
                device = setup_distributed_from_env()
                model = ToyModel().to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
                trainer = Trainer(
                    model,
                    optimizer,
                    loss_fn,
                    metrics={"mae": mae},
                    batch_size=2,
                    epochs=1,
                    output_dir=out_dir,
                    streaming=False,
                )
                step_result = trainer.step()
                test_result = trainer.test()
                payload = {
                    "rank": rank,
                    "train_samples": step_result["train"]["num_samples"],
                    "train_batches": step_result["train"]["num_batches"],
                    "train_loss": step_result["train"]["loss"],
                    "train_mae": step_result["train"]["metrics"]["mae"],
                    "val_samples": step_result["val"]["num_samples"],
                    "val_batches": step_result["val"]["num_batches"],
                    "val_loss": step_result["val"]["loss"],
                    "val_mae": step_result["val"]["metrics"]["mae"],
                    "test_samples": test_result["num_samples"],
                    "test_batches": test_result["num_batches"],
                    "test_loss": test_result["loss"],
                    "test_mae": test_result["metrics"]["mae"],
                    "progress": [
                        {
                            "desc": str(instance.desc),
                            "disable": instance.disable,
                            "postfixes": instance.postfixes,
                            "updates": instance.updates,
                        }
                        for instance in FakeTqdm.instances
                        if str(instance.desc).startswith(("train", "val", "test"))
                    ],
                }
                (out_dir / f"rank{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
            finally:
                if trainer is not None:
                    trainer.close()
                cleanup_distributed()
            """
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["CR_TRAIN_DIST_BACKEND"] = "gloo"
    env["CR_TRAIN_TEST_OUT"] = os.fspath(output_dir)
    env["OMP_NUM_THREADS"] = "1"
    src_path = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = os.fspath(src_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            os.fspath(script_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    records = [
        json.loads((output_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    records_by_rank = {record["rank"]: record for record in records}
    for record in records:
        assert record["train_samples"] == 8
        assert record["train_batches"] == 4
        assert record["train_loss"] == pytest.approx(6.375)
        assert record["train_mae"] == pytest.approx(2.25)
        assert record["val_samples"] == 8
        assert record["val_batches"] == 4
        assert record["val_loss"] == pytest.approx(53.875)
        assert record["val_mae"] == pytest.approx(7.25)
        assert record["test_samples"] == 8
        assert record["test_batches"] == 4
        assert record["test_loss"] == pytest.approx(151.375)
        assert record["test_mae"] == pytest.approx(12.25)

    rank0_progress = {entry["desc"]: entry for entry in records_by_rank[0]["progress"]}
    assert rank0_progress["train 1/1"]["updates"] == [1, 1]
    assert "loss: 4.1250" in rank0_progress["train 1/1"]["postfixes"][0]
    assert "mae: 1.7500" in rank0_progress["train 1/1"]["postfixes"][0]
    assert "loss: 6.3750" in rank0_progress["train 1/1"]["postfixes"][-1]
    assert rank0_progress["test"]["updates"] == [1, 1]
    assert "loss: 139.1250" in rank0_progress["test"]["postfixes"][0]
    assert "mae: 11.7500" in rank0_progress["test"]["postfixes"][0]
    assert "loss: 151.3750" in rank0_progress["test"]["postfixes"][-1]
    assert all(entry["disable"] for entry in records_by_rank[1]["progress"])

    metric_records = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_record = next(record for record in metric_records if record["kind"] == "train_epoch")
    val_record = next(record for record in metric_records if record["kind"] == "validation")
    test_record = next(record for record in metric_records if record["kind"] == "test")

    assert train_record["num_samples"] == 8
    assert train_record["num_batches"] == 4
    assert train_record["loss"] == pytest.approx(6.375)
    assert train_record["metrics"]["mae"] == pytest.approx(2.25)
    assert val_record["num_samples"] == 8
    assert val_record["loss"] == pytest.approx(53.875)
    assert test_record["num_samples"] == 8
    assert test_record["num_batches"] == 4
    assert test_record["loss"] == pytest.approx(151.375)
    assert test_record["metrics"]["mae"] == pytest.approx(12.25)
