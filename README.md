# cr-train

English | [한국어](./README.ko.md)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

A drop-in training toolkit for [SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/) cloud removal experiments.
Bring your own PyTorch model -- cr-train handles deterministic streaming, preprocessing, checkpointing, and progress tracking.

---

## Installation

```bash
# Recommended
uv add git+https://github.com/smturtle2/cr-train.git

# Or with pip
pip install git+https://github.com/smturtle2/cr-train.git
```

> Hugging Face rate limits apply. Set `export HF_TOKEN=your_token` for higher throughput.

## Quick Start

```python
import torch
from torch import nn
from cr_train import MAE, Trainer, TrainerConfig, build_sen12mscr_loaders


# 1. Define your model -- receives (sar, cloudy) as positional args
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(15, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 13, 1),
        )

    def forward(self, sar, cloudy):
        return self.net(torch.cat([sar, cloudy], dim=1))


# 2. Create data loaders (streams from Hugging Face, no local download)
train_loader, val_loader, test_loader = build_sen12mscr_loaders(batch_size=4)

# 3. Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)

trainer = Trainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    criterion=nn.MSELoss(),
    metrics=[MAE()],
    config=TrainerConfig(max_epochs=5, checkpoint_dir="checkpoints"),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

for epoch in trainer.step():
    print(f"Epoch {epoch['epoch']}  "
          f"train_loss={epoch['train']['loss']:.4f}  "
          f"val_loss={epoch['val']['loss']:.4f}")

print(trainer.test())
```

```bash
uv run my_train.py
```

For more examples: [`examples/minimal_train.py`](./examples/minimal_train.py) (CLI with all options), [`examples/colab_quickstart.ipynb`](./examples/colab_quickstart.ipynb) (Colab notebook).

---

## How It Works

```
Hugging Face (parquet shards)
  │
  ▼
build_sen12mscr_loaders()          ← scene-level split, streaming decode, preprocessing
  │
  ├── train_loader                 ← resharded + shuffled per epoch
  ├── val_loader                   ← fixed order
  └── test_loader                  ← fixed order
        │
        ▼
    Trainer.step()                 ← train/val loop, progress bars, auto-checkpointing
        │
        ▼
    Trainer.test()                 ← final evaluation
```

**Data pipeline.** Parquet shards are streamed directly from Hugging Face -- no local download required. Each sample is decoded to CHW tensors (2x256x256 SAR, 13x256x256 optical) and normalized on the fly.

**Scene isolation.** Scenes are assigned to train/val/test before any shuffling. No scene appears in multiple splits.

**Deterministic ordering.** Given the same `seed`, batch order is fully reproducible. The trainer calls `set_epoch()` on each epoch so shuffle order changes deterministically.

**Checkpointing.** When `checkpoint_dir` is set, `last.pt` and `epoch-NNNN.pt` are saved automatically after each epoch. Checkpoints include model, optimizer, scheduler (if provided), and full RNG state for exact resumption.

---

## Batch Format

Every batch is a dictionary with this structure:

```python
{
    "inputs": [
        Tensor,  # [B, 2, 256, 256]   SAR backscatter, float32 [0, 1]
        Tensor,  # [B, 13, 256, 256]  cloudy Sentinel-2, float32 [0, 1]
    ],
    "target": Tensor,      # [B, 13, 256, 256]  cloud-free Sentinel-2, float32 [0, 1]
    "metadata": {
        "season": list[str],        # "spring" | "summer" | "fall" | "winter"
        "scene":  list[str],        # scene ID
        "patch":  list[str],        # patch ID within scene
        "source_shard": list[str],  # e.g. "spring/scene_1.parquet"
    },
}
```

`inputs` is a list of `[sar, cloudy]`. The trainer unpacks it as `model(sar, cloudy)` -- parameter names in your `forward()` can be anything.

**Preprocessing:**

| Band | Raw range | Preprocessing | Output |
|------|-----------|---------------|--------|
| Optical (cloudy, target) | int16 [0, 10000] | `clip(0, 10000) / 10000` | float32 [0, 1] |
| SAR | float32 [-25, 0] dB | `clip(-25, 0)` then `(x + 25) / 25` | float32 [0, 1] |

---

## API Reference

### `build_sen12mscr_loaders(batch_size, **kwargs)`

Returns `(train_loader, val_loader, test_loader)`. Only train is shuffled; val/test maintain fixed order.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | required | Samples per batch |
| `seed` | `int` | `0` | Seed for shuffle order and worker init |
| `split` | `str` | `"official"` | `"official"` (author splits: 155/10/10 scenes) or `"seeded_scene"` (80/10/10 stratified by season) |
| `shuffle_buffer_size` | `int` | `16` | In-memory shuffle buffer for training |
| `num_workers` | `int \| None` | `None` | Auto: train gets `min(2, cpu_count//6)` workers, val/test get 0 |
| `pin_memory` | `bool` | `False` | Pin tensors for faster GPU transfer |
| `timeout` | `float` | `0.0` | Worker batch wait timeout in seconds; only applied to stages with `num_workers > 0` |
| `prefetch_factor` | `int \| None` | `None` | Auto `2` when workers > 0 |
| `persistent_workers` | `bool \| None` | `None` | Auto `False`; opt in when worker reuse is known to be stable |
| `io_profile` | `str` | `"smooth"` | `"smooth"` (light readahead) or `"conservative"` (fully synchronous) |

### `TrainerConfig`

Immutable (`frozen=True`) configuration for the training loop.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epochs` | `int` | `1` | Training epochs |
| `train_max_batches` | `int \| None` | `None` | Cap training batches per epoch |
| `val_max_batches` | `int \| None` | `None` | Cap validation batches per epoch |
| `test_max_batches` | `int \| None` | `None` | Cap test batches |
| `checkpoint_dir` | `str \| Path \| None` | `None` | Checkpoint directory (auto-created); `None` disables saving |
| `show_progress` | `bool \| None` | `None` | Progress bars; `None` = `True` |

### `Trainer`

```python
Trainer(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: (outputs, target) -> Tensor,
    metrics: [MAE(), ...] | None,
    config: TrainerConfig,
    train_loader,
    val_loader=None,
    test_loader=None,
    scheduler=None,            # e.g. torch.optim.lr_scheduler.*
)
```

Device is inferred from model parameters. All batch data is moved to that device automatically. Metrics are averaged per-sample (not per-batch), so the last smaller batch does not skew results.

#### `trainer.step(**overrides) -> Iterator[dict]`

Runs train/val epochs. Yields one record per epoch:

```python
{
    "epoch": 1,           # 1-indexed
    "global_step": 120,   # cumulative optimizer steps
    "train": {"loss": 0.0512, "mae": 0.0321},
    "val":   {"loss": 0.0498, "mae": 0.0315},  # {} if no val_loader
}
```

If a `scheduler` is provided, `scheduler.step()` is called after each epoch. Optional overrides: `max_epochs`, `train_max_batches`, `val_max_batches`.

#### `trainer.test(**overrides) -> dict[str, float]`

Returns `{"loss": ..., "mae": ...}`. Optional override: `max_batches`.

#### `trainer.save_checkpoint(filename) -> Path | None`

Saves to `checkpoint_dir/filename`. Returns `None` if checkpointing is disabled.

#### `trainer.load_checkpoint(path)`

Restores all state from a checkpoint file for exact training resumption.

### `MAE`

Built-in L1 loss metric. Usage: `metrics=[MAE()]`.

---

## Reproducibility

| Guarantee | Mechanism |
|-----------|-----------|
| No scene leakage | Scene-level split before any shuffling |
| Deterministic batches | Same `seed` + same settings = identical order |
| Per-epoch shuffle | `set_epoch()` propagated automatically (train only) |
| Exact resumption | Checkpoint captures model, optimizer, scheduler, and full RNG state |
| Pinned dataset version | Dataset revision is hardcoded |

The **official split** comes from the [authors' supplementary material](https://patricktum.github.io/cloud_removal/sen12mscr/) and is bundled in [`official_scene_splits.csv`](./src/cr_train/resources/official_scene_splits.csv).

---

## Project Structure

```
cr-train/
├── src/cr_train/
│   ├── __init__.py         # public API
│   ├── trainer.py          # training loop, checkpointing, progress
│   ├── data.py             # streaming dataset, scene splits, preprocessing
│   └── runtime.py          # parquet I/O tuning
├── examples/
│   ├── minimal_train.py    # full CLI training script
│   └── colab_quickstart.ipynb
├── tests/
├── scripts/
│   └── refresh_official_scene_splits.py
└── pyproject.toml
```

## License

[MIT](./LICENSE)
