# cr-train

English | [한국어](./README.ko.md)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Model-agnostic SEN12MS-CR training infrastructure for one job: reliable streaming experiments on [`Hermanni/sen12mscr`](https://huggingface.co/datasets/Hermanni/sen12mscr) with deterministic scene-level splitting and reproducible batches.

## What Changed

This README describes the current breaking API:

- no `DataModule`
- no `step_fn`
- no scheduler hook in `Trainer`
- data is built directly with `build_sen12mscr_dataset(...)` and `build_sen12mscr_dataloader(...)`
- training runs through `for history in trainer.step(): ...`

## Why this exists

SEN12MS-CR on Hugging Face is convenient, but the mirror exposes a single streaming split and stores `sar`, `cloudy`, and `target` as raw bytes. In practice you usually need more:

- streaming-only access end to end
- deterministic `train / val / test` splits
- scene-isolated evaluation without cross-scene leakage
- internal bytes-to-tensor decode
- reproducible batching
- a trainer that stays neutral to model architecture and loss design

`cr-train` packages those pieces into one reusable Python module.

## Features

- Always streaming. The data path uses Hugging Face streaming datasets end to end.
- Scene-isolated splits. Custom splits are resolved at `season/scene parquet shard` granularity.
- Official split support. A bundled manifest is derived from the authors' official supplementary `splits.csv`.
- Internal bytes decode. `sar`, `cloudy`, and `target` are decoded inside the dataset with shape-aware tensor restoration.
- Standard sample schema. Every sample and batch uses `inputs / target / metadata`.
- Shuffle discipline. Training shuffle always does `reshard() -> shuffle(seed, buffer_size)`.
- Reproducibility-first defaults. Same seed, split strategy, and loader topology give the same batches.
- Partial-epoch control. Training, validation, and test all support capped batch counts through `TrainerConfig`.
- Explicit device ownership. Move the model to its final device before creating the optimizer.

## Installation

```bash
uv sync
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
    LoaderConfig,
    SEN12MSCRDataConfig,
    ShuffleConfig,
    Trainer,
    TrainerConfig,
    build_sen12mscr_dataloader,
)


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
        return self.net(torch.cat([sar.float(), cloudy.float()], dim=1))


class FloatMSELoss(nn.Module):
    def forward(self, outputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(outputs, target.float())


data_config = SEN12MSCRDataConfig(
    split_strategy="seeded_scene",
    seed=7,
    loader=LoaderConfig(batch_size=4),
    shuffle=ShuffleConfig(buffer_size=16, reshard_num_shards=16),
)

train_loader = build_sen12mscr_dataloader("train", data_config)
val_loader = build_sen12mscr_dataloader("val", data_config)
test_loader = build_sen12mscr_dataloader("test", data_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyCloudRemovalNet().to(device)

# Move the model first, then build the optimizer from that moved model.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = FloatMSELoss()
metrics = {
    "mae": lambda outputs, target: F.l1_loss(outputs, target.float()),
}

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    metrics=metrics,
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
    print(history)

print(trainer.test())
```

Run the bundled example:

```bash
uv run python examples/minimal_train.py --epochs 1 --train-max-batches 8 --val-max-batches 2
```

## Data Contract

The standard schema for every decoded sample and every collated batch is:

```python
{
    "inputs": ...,
    "target": ...,
    "metadata": ...,
}
```

For SEN12MS-CR specifically:

- `inputs["sar"]`: SAR tensor
- `inputs["cloudy"]`: cloudy optical tensor
- `target`: cloud-free optical target tensor
- `metadata`: scene and decode metadata such as `season`, `scene`, `patch`, and `source_shard`

That means trainer-facing code should work against `outputs` and `target`, while the model receives `inputs`. For mapping inputs, `Trainer` calls `model(**inputs)`.

## Trainer Contract

`Trainer` takes:

- `model`
- `optimizer`
- `criterion`
- `metrics`
- `config`
- `train_loader`
- optional `val_loader`
- optional `test_loader`

The execution contract is intentionally simple:

- `criterion` is called as `criterion(outputs, target)`
- `metrics` is a mapping of `name -> callable(outputs, target)`
- training uses `for history in trainer.step(): ...`
- evaluation uses `trainer.test()`

There is no public `step_fn`, `DataModule`, or scheduler API in `Trainer`.

## Data Builders

```python
from cr_train import (
    SEN12MSCRDataConfig,
    build_sen12mscr_dataset,
    build_sen12mscr_dataloader,
)
```

Public builders:

- `SEN12MSCRDataConfig`
- `build_sen12mscr_dataset(stage, config, transform=None)`
- `build_sen12mscr_dataloader(stage, config, transform=None, collate_fn=None)`

Stages are `train`, `val`, and `test`.

Use `transform` when you want per-sample dataset transforms. Use `collate_fn` when you want custom batching on top of the standard sample schema.

## Data Guarantees

### 1. Streaming only

No map-style preprocessing layer is introduced. The module consumes the Hugging Face dataset through streaming iterables.

### 2. Scene-isolated splitting

The default custom split strategy is `seeded_scene`. It never splits after `reshard()`, so one original `season/scene parquet shard` belongs to exactly one of `train`, `val`, or `test`.

### 3. Internal tensor restoration

Each sample is restored inside the dataset:

- `sar`: decoded from bytes using `sample["dtype"]` and `sample["sar_shape"]`
- `cloudy`: decoded from bytes as `int16` using `sample["opt_shape"]`
- `target`: decoded from bytes as `int16` using `sample["opt_shape"]`

Default tensor layout is `channels_first`.

### 4. Shuffle invariant

Training shuffle obeys this fixed order:

```text
scene split -> streaming dataset -> reshard() -> shuffle(seed, buffer_size) -> batch
```

This keeps the split boundary scene-safe while still improving train-time mixing.

### 5. Reproducibility contract

Default loader settings are intentionally conservative:

- `num_workers=0`
- `in_order=True`
- `persistent_workers=False`

With the same seed, split strategy, batch size, and loader topology, you get the same batches. If you raise worker count or prefetching, treat that as a throughput-oriented mode rather than the strongest determinism path.

## Split Strategies

### `official`

Uses the authors' official supplementary split metadata, normalized into a bundled scene-level manifest at [`src/cr_train/resources/official_scene_splits.csv`](./src/cr_train/resources/official_scene_splits.csv).

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

SEN12MS-CR samples are large enough that aggressive streaming buffers can consume host memory very quickly.

## API Surface

```python
from cr_train import (
    LoaderConfig,
    SEN12MSCRDataConfig,
    ShuffleConfig,
    SplitRatios,
    Trainer,
    TrainerConfig,
    build_sen12mscr_dataloader,
    build_sen12mscr_dataset,
)
```

Key knobs:

- `SEN12MSCRDataConfig.split_strategy`: `official` or `seeded_scene`
- `SEN12MSCRDataConfig.seed`: global split and shuffle seed
- `SEN12MSCRDataConfig.loader`: `LoaderConfig(...)`
- `SEN12MSCRDataConfig.shuffle`: `ShuffleConfig(...)`
- `ShuffleConfig.buffer_size`: streaming shuffle buffer size
- `ShuffleConfig.reshard_num_shards`: reshard target before shuffle
- `LoaderConfig.batch_size`: batch size
- `TrainerConfig.max_epochs`: total epochs yielded by `trainer.step()`
- `TrainerConfig.train_max_batches`, `val_max_batches`, `test_max_batches`: partial-epoch caps
- `TrainerConfig.checkpoint_dir`: optional checkpoint directory

Runtime note:

- Hugging Face parquet streaming is bootstrapped internally when the dataset loader is created.
- The shutdown-crash workaround stays internal; there is no public runtime setup step.

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

Install dev and test dependencies:

```bash
uv sync --dev
```

Run the test suite:

```bash
uv run pytest
```

Refresh the bundled official scene manifest from the authors' supplementary release:

```bash
uv run python scripts/refresh_official_scene_splits.py
```

## Status

This repository is intentionally narrow: it is a training module for SEN12MS-CR, not a model zoo. The abstraction boundary is the data pipeline and training loop, not the network architecture.
