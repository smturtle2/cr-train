from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table
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


def _metric_columns(rows: Sequence[tuple[str, Mapping[str, float]]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for _, metrics in rows:
        for name in metrics:
            if name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


def _metrics_table(title: str, rows: Sequence[tuple[str, Mapping[str, float]]]) -> Table:
    columns = _metric_columns(rows)
    table = Table(title=title, header_style="bold cyan")
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
    title: str,
    stage_metrics: Sequence[tuple[str, Mapping[str, float]]],
) -> None:
    rows = [(stage, metrics) for stage, metrics in stage_metrics if metrics]
    if not rows:
        return
    console.print(_metrics_table(title, rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SEN12MS-CR streaming training example.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-max-batches", type=int, default=10)
    parser.add_argument("--val-max-batches", type=int, default=2)
    parser.add_argument("--test-max-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=("official", "seeded_scene"), default="official")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    args = parser.parse_args()

    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        args.batch_size,
        seed=args.seed,
        split=args.split,
        shuffle_buffer_size=16,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    console = Console()
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
        _print_summary(
            console,
            f"Epoch {history['epoch']} Summary (global step {history['global_step']})",
            (
                ("train", history["train"]),
                ("val", history["val"]),
            ),
        )
    _print_summary(console, "Test Summary", (("test", trainer.test()),))


if __name__ == "__main__":
    main()
