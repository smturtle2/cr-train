# cr-train

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Model-agnostic SEN12MS-CR training infrastructure built for one job: reliable streaming experiments on [`Hermanni/sen12mscr`](https://huggingface.co/datasets/Hermanni/sen12mscr) with deterministic scene-level splitting and reproducible batches.

## Why this exists

SEN12MS-CR on Hugging Face is convenient, but the mirror exposes a single streaming split and stores `sar`, `cloudy`, and `target` as raw bytes. For training work you usually need more than that:

- a dataset that decodes bytes into tensors internally
- deterministic `train / val / test` splits
- streaming-only data access
- scene-isolated evaluation without cross-scene leakage
- a trainer that does not hardcode a model, loss, optimizer, or scheduler
- the ability to run full epochs or capped-batch epochs

`cr-train` packages those pieces into one reusable Python module.

## Features

- Always streaming. The data path uses Hugging Face streaming datasets end-to-end.
- Scene-isolated splits. Custom splits are resolved at `season/scene parquet shard` granularity.
- Official split support. A vendored scene-level manifest is derived from the authors’ official supplementary `splits.csv`.
- Internal bytes decode. `sar`, `cloudy`, and `target` are decoded inside the custom dataset with shape-aware tensor restoration.
- Shuffle discipline. If training shuffle is enabled, the pipeline always does `reshard()` before `shuffle(seed, buffer_size)`.
- Strict reproducibility mode. Same seed + same split + same loader topology gives the same batches.
- Partial-epoch training. Train, validation, and test loops accept `max_batches`.
- Trainer neutrality. Model, `step_fn`, optimizer factory, and optional scheduler factory are injected.

## Installation

```bash
uv sync --dev
```

Optional but recommended for higher Hugging Face rate limits:

```bash
export HF_TOKEN=your_token
```

## Quick Start

```python
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
            nn.Conv2d(64, 13, kernel_size=1),
        )

    def forward(self, sar: torch.Tensor, cloudy: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([sar.float(), cloudy.float()], dim=1))


def step_fn(model: nn.Module, batch: dict[str, torch.Tensor], stage: str) -> StepResult:
    prediction = model(batch["sar"], batch["cloudy"])
    target = batch["target"].float()
    loss = F.mse_loss(prediction, target)
    return StepResult(loss=loss, metrics={"mae": F.l1_loss(prediction, target)})


datamodule = SEN12MSCRDataModule(
    DataModuleConfig(
        split_strategy="seeded_scene",
        seed=7,
        loader=LoaderConfig(batch_size=4),
        shuffle=ShuffleConfig(buffer_size=16, reshard_num_shards=16),
    )
)

trainer = Trainer(
    model=TinyCloudRemovalNet(),
    datamodule=datamodule,
    step_fn=step_fn,
    optimizer_factory=lambda model: torch.optim.AdamW(model.parameters(), lr=1e-4),
    checkpoint_dir="artifacts/checkpoints",
)

history = trainer.fit(max_epochs=2, train_max_batches=10, val_max_batches=2)
test_metrics = trainer.test(test_max_batches=2)
print(history)
print(test_metrics)
```

Run the bundled example:

```bash
uv run python examples/minimal_train.py --epochs 1 --train-max-batches 8 --val-max-batches 2
```

## Data Guarantees

### 1. Streaming only

No map-style preprocessing layer is introduced. The module consumes the Hugging Face dataset through streaming iterables.

### 2. Scene-isolated splitting

The default custom split strategy is `seeded_scene`. It never splits after `reshard()`. That means one original `season/scene parquet shard` belongs to exactly one of `train`, `val`, or `test`.

### 3. Internal tensor restoration

Each sample is restored inside the custom dataset:

- `sar`: decoded from bytes using `sample["dtype"]` and `sample["sar_shape"]`
- `cloudy`: decoded from bytes as `int16` using `sample["opt_shape"]`
- `target`: decoded from bytes as `int16` using `sample["opt_shape"]`

Default tensor layout is `channels_first`.

### 4. Shuffle invariant

Training shuffle obeys this fixed order:

```text
scene split -> streaming dataset -> reshard() -> shuffle(seed, buffer_size) -> batch
```

This is intentional. The split boundary stays scene-safe, while row-group reshuffling still improves train-time mixing.

### 5. Reproducibility contract

The default loader profile is `strict_repro`:

- `num_workers=0`
- `in_order=True`
- `persistent_workers=False`

With the same seed, split strategy, batch size, and loader topology, you get the same batches.  
If you need higher throughput, switch the loader profile and worker count explicitly, but treat that as a performance mode rather than the strongest determinism path.

## Split Strategies

### `official`

Uses the authors’ official supplementary split metadata, normalized into a bundled scene-level manifest at [`src/cr_train/resources/official_scene_splits.csv`](./src/cr_train/resources/official_scene_splits.csv).

Source provenance:

- official dataset page: <https://patricktum.github.io/cloud_removal/sen12mscr/>
- supplementary folder: <https://u.pcloud.link/publink/show?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV>
- current split file: <https://api.pcloud.com/getpubtextfile?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV&fileid=57823192235>

### `seeded_scene`

Builds deterministic custom splits from the full scene catalog using season-stratified shuffling and configurable ratios.

Default ratio:

```python
SplitRatios(train=0.8, val=0.1, test=0.1)
```

Default shuffle config is intentionally conservative for this dataset:

```python
ShuffleConfig(buffer_size=16, reshard_num_shards=1024)
```

SEN12MS-CR samples are large enough that aggressive streaming buffers can consume a lot of host memory very quickly.

## API Surface

```python
from cr_train import (
    DataModuleConfig,
    LoaderConfig,
    SEN12MSCRDataModule,
    ShuffleConfig,
    SplitRatios,
    StepResult,
    Trainer,
)
```

Key knobs:

- `DataModuleConfig.split_strategy`: `official` or `seeded_scene`
- `DataModuleConfig.seed`: global split and shuffle seed
- `ShuffleConfig.buffer_size`: streaming shuffle buffer size
- `ShuffleConfig.reshard_num_shards`: reshard target before shuffle
- `LoaderConfig.batch_size`: batch size
- `Trainer.fit(train_max_batches=..., val_max_batches=...)`: partial epochs
- `Trainer.test(test_max_batches=...)`: capped evaluation

## Project Layout

```text
src/cr_train/data.py                    # streaming dataset, split logic, dataloaders
src/cr_train/trainer.py                 # model-agnostic trainer
src/cr_train/resources/official_scene_splits.csv
examples/minimal_train.py               # minimal end-to-end usage
scripts/refresh_official_scene_splits.py
tests/                                  # reproducibility and trainer tests
```

## Development

Run the test suite:

```bash
uv run pytest
```

Refresh the bundled official scene manifest from the authors’ supplementary release:

```bash
uv run python scripts/refresh_official_scene_splits.py
```

## Status

This repository is intentionally narrow: it is a training module for SEN12MS-CR, not a model zoo. The abstraction boundary is the training loop and data pipeline, not the network architecture.
