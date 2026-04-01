from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from cr_train import Trainer


class FusionBaseline(nn.Module):
    """Small baseline for SAR + cloudy optical to target optical regression."""

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

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        fused = torch.cat([batch["sar"], batch["cloudy"]], dim=1)
        return self.head(self.stem(fused))


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
        description="Run SEN12MS-CR training with the canonical block cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-train-samples",
        type=parse_max_samples,
        default=2048,
        help="Requested train rows; rounded up to 16-row blocks internally, or 'none'/'full'.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=parse_max_samples,
        default=256,
        help="Requested validation rows; rounded up to 16-row blocks internally, or 'none'/'full'.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=parse_max_samples,
        default=256,
        help="Requested test rows; rounded up to 16-row blocks internally, or 'none'/'full'.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sample-selection seed over the canonical block stream.")
    parser.add_argument("--dataset-seed", type=int, default=None, help="Canonical dataset-stream shuffle seed.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--output-dir", default="runs/sen12mscr-example")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-channels", type=int, default=64)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        dataset_seed=args.dataset_seed,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
    )

    for _ in range(args.epochs):
        print(json.dumps(trainer.step(), ensure_ascii=True))

    print(json.dumps({"test": trainer.test()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
