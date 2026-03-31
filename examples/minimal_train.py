from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table
from torch import nn

from cr_train import MAE, Trainer, TrainerConfig, build_loaders


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


def _metric_columns(rows: Sequence[tuple[str, Mapping[str, float]]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for _, metrics in rows:
        for name in metrics:
            if name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


def _metrics_table(rows: Sequence[tuple[str, Mapping[str, float]]]) -> Table:
    columns = _metric_columns(rows)
    table = Table(header_style="bold cyan")
    table.add_column("stage")
    for name in columns:
        table.add_column(name, justify="right")
    for stage, metrics in rows:
        table.add_row(
            stage,
            *(f"{metrics[name]:.4f}" if name in metrics else "-" for name in columns),
        )
    return table


def _print_summary(
    console: Console,
    stage_metrics: Sequence[tuple[str, Mapping[str, float]]],
) -> None:
    rows = [(stage, metrics) for stage, metrics in stage_metrics if metrics]
    if not rows:
        return
    console.print(_metrics_table(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SEN12MS-CR training example.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-max-samples", type=int, default=400)
    parser.add_argument("--val-max-samples", type=int, default=80)
    parser.add_argument("--test-max-samples", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    args: Any = parser.parse_args()
    console = Console()

    train_loader, val_loader, test_loader = build_loaders(
        args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
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
            train_max_samples=args.train_max_samples,
            val_max_samples=args.val_max_samples,
            test_max_samples=args.test_max_samples,
            checkpoint_dir=args.checkpoint_dir,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    for history in trainer.step():
        _print_summary(
            console,
            (
                ("train", history["train"]),
                ("val", history["val"]),
            ),
        )
    _print_summary(console, (("test", trainer.test()),))


if __name__ == "__main__":
    main()
