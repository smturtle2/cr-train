"""SEN12MS-CR training example built around ``Trainer``.

Create a model, optimizer, loss, and custom scheduler, then let ``Trainer``
handle dataset access, cache warmup, dataloaders, and metrics.

Usage:
    uv run python examples/train_sen12mscr.py \\
      --max-train-samples 2048 \\
      --max-val-samples 256 \\
      --batch-size 4 \\
      --accum-steps 4 \\
      --grad-clip-norm 1.0 \\
      --epochs 2 \\
      --scheduler warmup-cosine \\
      --scheduler-timing after_validation \\
      --warmup-epochs 1 \\
      --train-crop-size 128 \\
      --train-random-flip \\
      --train-random-rot90 \\
      --output-dir runs/sen12mscr-example

Output:
    Each epoch prints a human-readable summary (loss, metrics, lr, elapsed time)
    and writes structured JSON to stdout for pipeline consumption.
    Call trainer.save_checkpoint() explicitly if you want a resumable snapshot.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F

from cr_train import Trainer


# --- Model ---

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


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Example epoch-based scheduler compatible with ``Trainer``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_epochs: int,
        warmup_epochs: int = 1,
        warmup_start_factor: float = 0.25,
        min_lr_scale: float = 0.1,
    ) -> None:
        if total_epochs <= 0:
            raise ValueError("total_epochs must be greater than zero")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be zero or greater")
        if not 0.0 < warmup_start_factor <= 1.0:
            raise ValueError("warmup_start_factor must be in the range (0, 1]")
        if not 0.0 <= min_lr_scale <= 1.0:
            raise ValueError("min_lr_scale must be in the range [0, 1]")

        self.total_epochs = total_epochs
        self.warmup_epochs = min(warmup_epochs, max(total_epochs - 1, 0))
        self.warmup_start_factor = warmup_start_factor
        self.min_lr_scale = min_lr_scale
        super().__init__(optimizer)

    def _scale_for_epoch(self, epoch_index: int) -> float:
        if self.total_epochs == 1:
            return 1.0
        if self.warmup_epochs > 0 and epoch_index < self.warmup_epochs:
            if self.warmup_epochs == 1:
                return 1.0
            warmup_progress = epoch_index / (self.warmup_epochs - 1)
            return self.warmup_start_factor + (1.0 - self.warmup_start_factor) * warmup_progress

        decay_denominator = max(1, self.total_epochs - self.warmup_epochs - 1)
        decay_progress = min(1.0, (epoch_index - self.warmup_epochs) / decay_denominator)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.min_lr_scale + (1.0 - self.min_lr_scale) * cosine_scale

    def get_lr(self) -> list[float]:
        scale = self._scale_for_epoch(self.last_epoch)
        return [base_lr * scale for base_lr in self.base_lrs]


# --- Argument parsing ---

def parse_max_samples(value: str) -> int | None:
    lowered = value.strip().lower()
    if lowered in {"none", "full"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("sample counts must be positive or 'none'")
    return parsed


def parse_non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_optional_positive_float(value: str) -> float | None:
    lowered = value.strip().lower()
    if lowered in {"none", "off"}:
        return None
    return parse_positive_float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SEN12MS-CR training with Trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-train-samples",
        type=parse_max_samples,
        default=None,
        help="Requested train rows; converted to block count with the fixed 64-row BLOCK_SIZE, or 'none'/'full'.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=parse_max_samples,
        default=None,
        help="Requested validation rows; converted to block count with the fixed 64-row BLOCK_SIZE, or 'none'/'full'.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=parse_max_samples,
        default=None,
        help="Requested test rows; converted to block count with the fixed 64-row BLOCK_SIZE, or 'none'/'full'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed controlling block selection and epoch-wise block/row shuffle order.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--accum-steps",
        type=parse_positive_int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer update.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--grad-clip-norm",
        type=parse_optional_positive_float,
        default=1.0,
        help="Max grad norm applied before each optimizer update, or 'none'/'off' to disable.",
    )
    parser.add_argument("--output-dir", default="runs/sen12mscr-example")
    parser.add_argument("--cache-dir", default="/dhdd/.cache/cr-train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument(
        "--scheduler",
        choices=("none", "warmup-cosine"),
        default="warmup-cosine",
        help="Optional epoch-based scheduler to attach to Trainer.",
    )
    parser.add_argument(
        "--scheduler-timing",
        choices=("after_validation", "before_optimizer_step", "after_optimizer_step"),
        default="after_validation",
        help=(
            "When Trainer should call scheduler.step(). "
            "Keep the bundled warmup-cosine example at the default epoch-based after_validation timing."
        ),
    )
    parser.add_argument(
        "--warmup-epochs",
        type=parse_non_negative_int,
        default=1,
        help="Warmup epochs for the custom warmup+cosine scheduler.",
    )
    parser.add_argument(
        "--min-lr-scale",
        type=float,
        default=0.1,
        help="Final learning-rate scale for the cosine tail.",
    )
    parser.add_argument(
        "--train-crop-size",
        type=parse_max_samples,
        default=128,
        help="Random train crop size, or 'none' to disable cropping.",
    )
    parser.add_argument(
        "--train-random-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable random train flips.",
    )
    parser.add_argument(
        "--train-random-rot90",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable random 90-degree train rotations.",
    )
    return parser.parse_args()


# --- Loss and metrics ---

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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
    min_lr_scale: float,
) -> WarmupCosineScheduler | None:
    if scheduler_name == "none":
        return None
    if scheduler_name != "warmup-cosine":
        raise ValueError(f"unsupported scheduler: {scheduler_name}")
    return WarmupCosineScheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=warmup_epochs,
        min_lr_scale=min_lr_scale,
    )


# --- Main ---

def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = FusionBaseline(hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_scale=args.min_lr_scale,
    )

    trainer = Trainer(
        model,
        optimizer,
        reconstruction_loss,
        metrics={"mae": mean_absolute_error},
        scheduler=scheduler,
        scheduler_timing=args.scheduler_timing,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        train_crop_size=args.train_crop_size,
        train_random_flip=args.train_random_flip,
        train_random_rot90=args.train_random_rot90,
        grad_clip_norm=args.grad_clip_norm,
    )

    # Training loop — Trainer prints epoch summaries automatically
    for _ in range(args.epochs):
        trainer.step()

    # Test evaluation — Trainer prints test summary automatically
    trainer.test()


if __name__ == "__main__":
    main()
