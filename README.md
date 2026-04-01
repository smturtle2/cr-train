# cr-train

HF-first SEN12MS-CR training module built around the Hugging Face dataset
[`Hermanni/sen12mscr`](https://huggingface.co/datasets/Hermanni/sen12mscr).

## Install

```bash
uv add git+https://github.com/your-org/cr-train
```

## Usage

Run the standalone example to launch an actual training job:

```bash
uv run python examples/train_sen12mscr.py \
  --max-train-samples 2048 \
  --max-val-samples 256 \
  --max-test-samples 256 \
  --batch-size 4 \
  --epochs 1 \
  --output-dir runs/sen12mscr-example
```

Prototype the `shuffle(buffer).take(N)` cache idea separately:

```bash
uv run python examples/benchmark_take_cache.py \
  --split train \
  --sample-sizes 64,256 \
  --buffer-sizes 256,1024 \
  --cache-dir runs/take-cache-bench \
  --clear-cache
```

The package API stays minimal:

```python
from cr_train import Trainer
import torch
from torch import nn
from torch.nn import functional as F


class FusionBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 13, kernel_size=1),
        )

    def forward(self, batch):
        fused = torch.cat([batch["sar"], batch["cloudy"]], dim=1)
        return self.body(fused)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionBaseline().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

trainer = Trainer(
    model,
    optimizer,
    lambda prediction, batch: F.l1_loss(prediction, batch["target"]),
    metrics={
        "mae": lambda prediction, batch: torch.mean(torch.abs(prediction - batch["target"])),
    },
    max_train_samples=2048,
    max_val_samples=256,
    max_test_samples=256,
    batch_size=4,
    epochs=2,
    seed=42,        # sample-selection seed
    dataset_seed=7, # canonical dataset-stream seed
    output_dir="runs/sen12mscr",
)

for _ in range(trainer.epochs):
    print(trainer.step())

print(trainer.test())
```

## Notes

- The first `step()` or `test()` warms local cache for `train`, `validation`, and
  `test` before any batch runs.
- Training and evaluation always read from the persistent local cache after warmup.
- Cache identity is based on `split + dataset_seed + shuffle_buffer_size`.
- `dataset_seed` fixes a canonical shuffled stream for each split.
- `seed` drives the block-selection planner over that fixed stream.
- `max_*_samples` is treated as requested row count, and the internal execution
  plan rounds up to whole 16-row blocks.
- The sampler scans the candidate window from left to right and raises take
  probability as the remaining window shrinks, so cache warmup tends to finish
  earlier than a uniform exact-`k` block draw.
- Warmup expands the canonical stream in block order, caches only missing blocks,
  and skips Hugging Face entirely when every selected block is already cached.
- The example script accepts `--max-*-samples none` (or `full`) to trigger
  full-split caching explicitly.
- Equal `seed` values keep the same block-selection membership, and train batch
  order still changes by epoch.
- Finished caches are never auto-deleted. Remove them manually from the cache
  directory if you want to reclaim disk space.
- `Trainer.step()` runs exactly one training epoch and shows running-average loss
  and metrics with batch-based `tqdm`.
- Cache warmup shows a short block-resolution `tqdm` plus one-line cache
  summaries instead of verbose startup logs.
