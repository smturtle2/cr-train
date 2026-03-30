from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
    # stage마다 metric 집합이 조금씩 달라도 한 표에서 같이 보여줄 수 있게 열을 합친다.
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


def _parse_num_workers(value: str) -> int | None:
    if value == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("num_workers must be 'auto' or a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("num_workers must be non-negative")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SEN12MS-CR training example.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-max-samples", type=int, default=400)
    parser.add_argument("--val-max-samples", type=int, default=80)
    parser.add_argument("--test-max-samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=("official", "seeded_scene"), default="official")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download the full dataset instead of streaming from Hugging Face",
    )
    parser.add_argument("--num-workers", type=_parse_num_workers, default=None, metavar="auto|INT")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--io-profile", choices=("smooth", "conservative"), default="smooth")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    args: Any = parser.parse_args()

    # 로컬 모듈 사용 예제 기준점이 되도록 loader 옵션을 CLI에서 바로 조절할 수 있게 둔다.
    train_loader, val_loader, test_loader = build_sen12mscr_loaders(
        args.batch_size,
        streaming=not args.no_streaming,
        seed=args.seed,
        split=args.split,
        shuffle_buffer_size=64,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        timeout=args.timeout,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        io_profile=args.io_profile,
    )

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCloudRemovalNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Trainer가 train/val/test 루프와 checkpoint, progress 출력을 모두 묶어서 관리한다.
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
        # epoch별 평균 metric만 따로 표로 다시 보여줘 실험 로그를 훑기 쉽게 만든다.
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
