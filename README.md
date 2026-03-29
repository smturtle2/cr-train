# cr-train

English | [한국어](./README.ko.md)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Model-agnostic SEN12MS-CR training utilities for deterministic streaming experiments.

## Features

- **Deterministic streaming** -- scene-level splits with reproducible batch ordering across epochs
- **Model-agnostic** -- works with any PyTorch `nn.Module`; just implement `forward(sar, cloudy)`
- **Auto-tuned I/O** -- worker count, prefetch, and parquet readahead configured automatically
- **Checkpoint management** -- full RNG state capture for exact resumption
- **Built-in progress** -- `tqdm.rich` stage bars with live loss/metric updates

## Project Structure

```
cr-train/
├── src/cr_train/           # Core package
│   ├── __init__.py         #   public API exports
│   ├── trainer.py          #   Trainer, TrainerConfig, MAE
│   ├── data.py             #   dataset loading & preprocessing
│   └── runtime.py          #   parquet I/O tuning
├── examples/
│   ├── minimal_train.py    # Reference training script
│   └── colab_quickstart.ipynb
├── tests/
├── scripts/
└── pyproject.toml
```

## Quick Start

```bash
git clone https://github.com/smturtle2/cr-train.git
cd cr-train
uv sync
```

Optional but recommended for higher Hugging Face rate limits:

```bash
export HF_TOKEN=your_token
```

Run the reference example:

```bash
uv run python examples/minimal_train.py --epochs 1 --train-max-batches 10 --val-max-batches 2
```

Or try the Colab notebook: [`examples/colab_quickstart.ipynb`](./examples/colab_quickstart.ipynb)

## Usage

This repository is meant to be cloned and used as a local training module.

1. Clone and install: `git clone ... && uv sync`
2. Run scripts from the repository root so the local `cr_train` package is importable.
3. Import from `cr_train` (not `src.cr_train`):

```python
from cr_train import Trainer, TrainerConfig, MAE, build_sen12mscr_loaders

train_loader, val_loader, test_loader = build_sen12mscr_loaders(batch_size=4)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=nn.MSELoss(),
    metrics=[MAE()],
    config=TrainerConfig(max_epochs=5),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

for history in trainer.step():
    print(history["train"], history["val"])

test_metrics = trainer.test()
```

For your own scripts, a common layout is:

- `examples/` -- runnable reference scripts
- `scripts/` -- one-off experiments
- Repository root -- quick experiments via `uv run python your_script.py`

The reference implementation is [`examples/minimal_train.py`](./examples/minimal_train.py).

## Public API

### `build_sen12mscr_loaders`

```python
build_sen12mscr_loaders(
    batch_size,
    *,
    seed=0,
    split="official",              # "official" | "seeded_scene"
    shuffle_buffer_size=16,
    num_workers=None,              # None = auto-tune
    pin_memory=False,
    prefetch_factor=None,
    persistent_workers=None,
    io_profile="smooth",           # "smooth" | "conservative"
) -> tuple[DataLoader, DataLoader, DataLoader]
```

Returns `(train_loader, val_loader, test_loader)`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_workers` | `None` | Notebook-safe auto mode: a small train-only worker pool |
| `io_profile` | `"smooth"` | Light parquet readahead without extra thread fan-out; use `"conservative"` for fully synchronous I/O |
| `persistent_workers` | `None` | Defaults to `False`; opt in explicitly if you want long-lived workers |

### `Trainer`

```python
Trainer(
    model,                         # nn.Module
    optimizer,                     # torch.optim.Optimizer
    criterion,                     # loss function
    metrics,                       # e.g. [MAE()]
    config,                        # TrainerConfig
    train_loader,
    val_loader=None,
    test_loader=None,
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `trainer.step()` | Iterator yielding per-epoch history dicts |
| `trainer.test()` | Evaluate on test loader, returns metrics dict |
| `trainer.save_checkpoint(path)` | Save model + optimizer + RNG state |
| `trainer.load_checkpoint(path)` | Restore from checkpoint |

**Progress bars:**

- Uses `tqdm.rich` by default; disable with `show_progress=False`.
- Each epoch prints a heading, then stage bars render underneath.
- Bars show `loading first batch...` initially, then update running `loss` and metrics per batch.
- `max_batches` stops cleanly without prefetching unused data.
- Auto loader defaults favor notebook safety: train gets a small background worker pool, val/test stay synchronous, and stage workers are not kept alive unless you opt in.

**Built-in metric:** `MAE` (Mean Absolute Error).

## Batch Contract

Each sample and collated batch follows this schema:

```python
{
    "inputs": {
        "sar":    Tensor,    # [B, 2, H, W]   SAR backscatter
        "cloudy": Tensor,    # [B, 13, H, W]  cloudy optical
    },
    "target":     Tensor,    # [B, 13, H, W]  cloud-free optical
    "metadata": {
        "season": list[str],
        "scene":  list[str],
        "patch":  list[str],
        ...
    },
}
```

`Trainer` forwards `inputs` to the model and applies `criterion(output, target)` plus each metric to `(output, target)`.

## Data Defaults

| Setting | Value |
|---------|-------|
| Split | `official` (175 scenes: 155 train / 10 val / 10 test) |
| Optical preprocessing | `clip(0, 10000) / 10000.0` -> float32 |
| SAR preprocessing | `clip(-25, 0)`, then `(x + 25) / 25` -> [0, 1] |
| Train pipeline | `reshard() -> shuffle(seed, buffer_size)` -> batch |
| Scene membership | Fixed across epochs; only train order changes deterministically |
| Dataset revision | Pinned for reproducibility |

## Reproducibility

- Scene-level splitting keeps train, validation, and test sets scene-isolated.
- The official split is bundled from the authors' supplementary metadata.
- With the same seed and loader settings, batch order is fully reproducible.

## Official Split Source

| Resource | Link |
|----------|------|
| Project page | <https://patricktum.github.io/cloud_removal/sen12mscr/> |
| Supplementary folder | <https://u.pcloud.link/publink/show?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV> |
| Direct `splits.csv` | <https://api.pcloud.com/getpubtextfile?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV&fileid=57823192235> |
| Bundled manifest | [`official_scene_splits.csv`](./src/cr_train/resources/official_scene_splits.csv) |
| Refresh script | [`refresh_official_scene_splits.py`](./scripts/refresh_official_scene_splits.py) |
