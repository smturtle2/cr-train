"""SEN12MS-CR training example built around ``Trainer``.

Create a model, optimizer, and loss, then let ``Trainer`` handle dataset access,
cache warmup, dataloaders, metrics, and checkpoints.

Usage:
    uv run python examples/train_sen12mscr.py \\
      --max-train-samples 2048 \\
      --max-val-samples 256 \\
      --batch-size 4 \\
      --epochs 2 \\
      --output-dir runs/sen12mscr-example

Output:
    Each epoch prints a human-readable summary (loss, metrics, throughput)
    and writes structured JSON to stdout for pipeline consumption.
    Checkpoints are saved to <output-dir>/epoch-NNNN.pt.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F

from cr_train import Trainer


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FusionBaseline(nn.Module):
    """Small baseline CNN for SAR + cloudy optical -> target optical regression.

    Input channels:  2 (SAR) + 13 (cloudy optical) = 15
    Output channels: 13 (cloud-free optical)
    """

    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(15, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 13, kernel_size=1),
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        return self.head(self.stem(torch.cat([sar, cloudy], dim=1)))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_max_samples(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"none", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("sample counts must be positive or 'none'")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SEN12MS-CR training with Trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-train-samples",
        type=parse_max_samples,
        default=2048,
        help="Requested train rows; converted to block count with the fixed 64-row BLOCK_SIZE, or 'none'/'full'.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=parse_max_samples,
        default=256,
        help="Requested validation rows; converted to block count with the fixed 64-row BLOCK_SIZE, or 'none'/'full'.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=parse_max_samples,
        default=256,
        help="Requested test rows; converted to block count with the fixed 64-row BLOCK_SIZE, or 'none'/'full'.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed controlling row-group order and block selection.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--output-dir", default="runs/sen12mscr-example")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-channels", type=int, default=64)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------

def resolve_device(device_name: str | None) -> torch.device:
    if device_name is not None:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reconstruction_loss(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.l1_loss(prediction, batch["target"])


def mean_absolute_error(prediction: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - batch["target"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = FusionBaseline(hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainer = Trainer(
        model,
        optimizer,
        reconstruction_loss,
        metrics={"mae": mean_absolute_error},
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )

    # Training loop — Trainer prints epoch summaries automatically
    for _ in range(args.epochs):
        trainer.step()

    # Test evaluation — Trainer prints test summary automatically
    trainer.test()


if __name__ == "__main__":
    main()
