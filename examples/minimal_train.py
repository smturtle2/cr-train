from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from cr_train import MAE, Trainer, TrainerConfig, build_sen12mscr_loaders


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
        return self.net(torch.cat([sar, cloudy], dim=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SEN12MS-CR streaming training example.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-max-batches", type=int, default=10)
    parser.add_argument("--val-max-batches", type=int, default=2)
    parser.add_argument("--test-max-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=("official", "seeded_scene"), default="official")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    args = parser.parse_args()

    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        args.batch_size,
        seed=args.seed,
        split=args.split,
        shuffle_buffer_size=16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCloudRemovalNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        metrics=[MAE()],
        config=TrainerConfig(
            max_epochs=args.epochs,
            train_max_batches=args.train_max_batches,
            val_max_batches=args.val_max_batches,
            test_max_batches=args.test_max_batches,
            checkpoint_dir=args.checkpoint_dir,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    for history in trainer.step():
        print(history)
    print(trainer.test())


if __name__ == "__main__":
    main()
