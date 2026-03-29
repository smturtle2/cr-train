# cr-train

English | [한국어](./README.ko.md)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Model-agnostic SEN12MS-CR training utilities for deterministic streaming experiments.

## Installation

```bash
uv sync
```

Colab-style onboarding notebook:

- [`examples/colab_quickstart.ipynb`](/home/smturtle2/projects/cr-train/examples/colab_quickstart.ipynb)

The notebook still uses `uv` for fast environment setup; Colab or Jupyter remains the notebook runtime.

Optional but recommended for higher Hugging Face rate limits:

```bash
export HF_TOKEN=your_token
```

## Quick Start

```python
from collections.abc import Mapping, Sequence

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


def metric_columns(rows: Sequence[tuple[str, Mapping[str, float]]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for _, metrics in rows:
        for name in metrics:
            if name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


def print_summary(
    console: Console,
    title: str,
    rows: Sequence[tuple[str, Mapping[str, float]]],
) -> None:
    visible_rows = [(stage, metrics) for stage, metrics in rows if metrics]
    if not visible_rows:
        return

    columns = metric_columns(visible_rows)
    table = Table(title=title, header_style="bold cyan")
    table.add_column("stage")
    for name in columns:
        table.add_column(name, justify="right")
    for stage, metrics in visible_rows:
        table.add_row(stage, *(f"{metrics[name]:.4f}" if name in metrics else "-" for name in columns))
    console.print(table)


train_loader, val_loader, test_loader = build_sen12mscr_loaders(
    batch_size=4,
)

console = Console()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCloudRemovalNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    metrics=[MAE()],
    config=TrainerConfig(
        max_epochs=2,
        train_max_batches=10,
        val_max_batches=2,
        test_max_batches=2,
        checkpoint_dir="artifacts/checkpoints",
    ),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

for history in trainer.step():
    print_summary(
        console,
        f"Epoch {history['epoch']} Summary (global step {history['global_step']})",
        (
            ("train", history["train"]),
            ("val", history["val"]),
        ),
    )

print_summary(console, "Test Summary", (("test", trainer.test()),))
```

## Public API

```python
build_sen12mscr_loaders(
    batch_size,
    *,
    seed=0,
    split="official",
    shuffle_buffer_size=16,
    num_workers=0,
    pin_memory=False,
    prefetch_factor=None,
    persistent_workers=False,
)
```

Returns `(train_loader, val_loader, test_loader)`.

```python
Trainer(
    model,
    optimizer,
    criterion,
    metrics,
    config,
    train_loader,
    val_loader=None,
    test_loader=None,
)
```

- `metrics` is a sequence of metric objects, for example `metrics=[MAE()]`.
- Built-in metric object: `MAE`.
- Training runs through `for history in trainer.step(): ...`.
- Evaluation runs through `trainer.test()`.
- `train`, `val`, and `test` progress bars use `tqdm.rich` by default; set `show_progress=False` to disable them.
- Each epoch prints a heading first, then stage bars render underneath it and stay visible after completion.
- Stage bars show `loading first batch...` before the first batch arrives, then update running `loss` and metrics on every batch.
- `num_workers`, `prefetch_factor`, and `persistent_workers` can be used to reduce first-batch latency and improve throughput on real training runs. `prefetch_factor` and `persistent_workers` require `num_workers > 0`.

## Batch Contract

Each decoded sample and collated batch uses the same schema:

```python
{
    "inputs": {
        "sar": ...,
        "cloudy": ...,
    },
    "target": ...,
    "metadata": ...,
}
```

`Trainer` forwards `inputs` to the model and applies `criterion(outputs, target)` plus each metric object to `(outputs, target)`.

## Data Defaults

- Default split is `official`.
- Optical preprocessing defaults to `clip(0, 10000)` and then `/ 10000.0` to `float32` for both `cloudy` and `target`.
- SAR preprocessing defaults to `clip(-25, 0)` and then `(x + 25) / 25` to map values into `[0, 1]`.
- The train pipeline keeps the invariant `reshard() -> shuffle(seed, buffer_size)` before batching.
- The dataset revision is pinned for reproducibility.

## Reproducibility

- Scene-level splitting keeps train, validation, and test scene-isolated.
- The official split is bundled from the authors' supplementary metadata.
- With the same seed and loader settings, batch order is reproducible.

## Official Split Source

- Official project page: <https://patricktum.github.io/cloud_removal/sen12mscr/>
- Supplementary folder: <https://u.pcloud.link/publink/show?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV>
- Direct `splits.csv`: <https://api.pcloud.com/getpubtextfile?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV&fileid=57823192235>
- Bundled normalized manifest: [`official_scene_splits.csv`](/home/smturtle2/projects/cr-train/src/cr_train/resources/official_scene_splits.csv)
- Refresh script: [`refresh_official_scene_splits.py`](/home/smturtle2/projects/cr-train/scripts/refresh_official_scene_splits.py)
