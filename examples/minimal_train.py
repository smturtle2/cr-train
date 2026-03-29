from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from cr_train import (
    DataModuleConfig,
    LoaderConfig,
    SEN12MSCRDataModule,
    ShuffleConfig,
    StepResult,
    Trainer,
)


class TinyCloudRemovalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 13, kernel_size=1),
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        optical = cloudy.float()
        fused = torch.cat([sar.float(), optical], dim=1)
        return self.net(fused)


def step_fn(model: nn.Module, batch: dict[str, torch.Tensor], stage: str) -> StepResult:
    prediction = model(batch["sar"], batch["cloudy"])
    target = batch["target"].float()
    loss = F.mse_loss(prediction, target)
    mae = F.l1_loss(prediction, target)
    return StepResult(loss=loss, metrics={"mae": mae})


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SEN12MS-CR streaming training example.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-max-batches", type=int, default=10)
    parser.add_argument("--val-max-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-strategy", choices=("official", "seeded_scene"), default="seeded_scene")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    args = parser.parse_args()

    datamodule = SEN12MSCRDataModule(
        DataModuleConfig(
            split_strategy=args.split_strategy,
            seed=args.seed,
            loader=LoaderConfig(batch_size=args.batch_size),
            shuffle=ShuffleConfig(buffer_size=16, reshard_num_shards=16),
        )
    )
    model = TinyCloudRemovalNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        datamodule=datamodule,
        step_fn=step_fn,
        checkpoint_dir=args.checkpoint_dir,
    )
    history = trainer.fit(
        max_epochs=args.epochs,
        train_max_batches=args.train_max_batches,
        val_max_batches=args.val_max_batches,
    )
    print(history)
    print(trainer.test(test_max_batches=1))


if __name__ == "__main__":
    main()
