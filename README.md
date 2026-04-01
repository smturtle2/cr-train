<h1 align="center">cr-train</h1>

<p align="center">
  <em>HuggingFace-first training module for satellite cloud removal on SEN12MS-CR</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python 3.12+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch 2.4+"></a>
  <a href="https://huggingface.co/datasets/Hermanni/sen12mscr"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hermanni%2Fsen12mscr-yellow" alt="Dataset"></a>
</p>

<p align="center">
  <b>English</b> | <a href="README.ko.md">한국어</a>
</p>

---

## Highlights

- **Single-class API** -- `Trainer` with `step()` + `test()`, nothing else to learn
- **Deterministic block sampling** -- two-seed system (`seed` + `dataset_seed`) for exact reproducibility
- **Smart cache warmup** -- downloads only missing blocks; skips HuggingFace entirely when cache is complete
- **Distributed training** -- automatic DDP wrapping, `DistributedSampler`, all-reduce metrics
- **JSONL experiment tracking** -- every epoch, validation, checkpoint, and startup event recorded to `metrics.jsonl`
- **Zero config data** -- streams directly from [`Hermanni/sen12mscr`](https://huggingface.co/datasets/Hermanni/sen12mscr); no manual download needed

---

## Quick Start

### Installation

```bash
uv add git+https://github.com/your-org/cr-train
```

### Minimal example

```python
from cr_train import Trainer
import torch
from torch import nn
from torch.nn import functional as F

class FusionBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 SAR channels + 13 cloudy optical channels = 15 input channels
        self.body = nn.Sequential(
            nn.Conv2d(15, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 13, 1),  # 13 target optical channels
        )

    def forward(self, sar, cloudy):
        return self.body(torch.cat([sar, cloudy], dim=1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionBaseline().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

trainer = Trainer(
    model, optimizer,
    loss=lambda pred, batch: F.l1_loss(pred, batch["target"]),
    metrics={"mae": lambda pred, batch: torch.mean(torch.abs(pred - batch["target"]))},
    max_train_samples=2048,
    max_val_samples=256,
    max_test_samples=256,
    batch_size=4,
    epochs=2,
    seed=42,
    dataset_seed=7,
    output_dir="runs/sen12mscr",
)

for _ in range(trainer.epochs):
    print(trainer.step())

print(trainer.test())
```

---

## Examples

### CLI training

Run the bundled training script with the built-in `FusionBaseline` model:

```bash
uv run python examples/train_sen12mscr.py \
  --max-train-samples 2048 \
  --max-val-samples 256 \
  --max-test-samples 256 \
  --batch-size 4 \
  --epochs 2 \
  --output-dir runs/sen12mscr-example
```

Pass `--max-train-samples none` (or `full`) to cache and train on the entire split.

### Sampling algorithm visualization

See how the block-selection bitmask is built step by step:

```bash
uv run python examples/bitmask_sampling_demo.py \
  --total-rows 107072 \
  --requested-rows 2048 \
  --seed 9
```

Output shows each block's take probability, the random draw, and a final bitmap of selected (`■`) vs. skipped (`□`) blocks.

---

## Architecture

```mermaid
flowchart TD
    HF["HuggingFace Hub<br/><code>Hermanni/sen12mscr</code>"]
    SHUFFLE["Streaming Shuffle<br/><code>dataset_seed</code>"]
    PLAN["Block Selection Planner<br/><code>seed</code> · sequential additive exact-k"]
    CACHE["Local Block Cache<br/>Arrow chunks · 16 rows/block"]
    DS["CachedBlockDataset"]
    DL["DataLoader<br/>DDP-aware · pinned memory"]
    TRAIN["Trainer.step() / test()"]
    OUT["Outputs<br/><code>metrics.jsonl</code> · <code>epoch-NNNN.pt</code>"]

    HF --> SHUFFLE --> PLAN --> CACHE
    CACHE --> DS --> DL --> TRAIN --> OUT
```

---

## API Reference

### `Trainer.__init__`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *(required)* | PyTorch model. `forward(sar, cloudy)` signature, returns a prediction tensor. |
| `optimizer` | `Optimizer` | *(required)* | Must be constructed from `model.parameters()`. |
| `loss` | `Callable` | *(required)* | `(prediction, batch) -> scalar tensor`. |
| `metrics` | `dict[str, Callable]` | `None` | `{"name": (prediction, batch) -> scalar}`. Logged per epoch. |
| `max_train_samples` | `int \| None` | `None` | Requested train rows. Rounded up to 16-row blocks. `None` = full split. |
| `max_val_samples` | `int \| None` | `None` | Same for validation. |
| `max_test_samples` | `int \| None` | `None` | Same for test. |
| `batch_size` | `int` | `4` | Batch size for all DataLoaders. |
| `epochs` | `int` | `1` | Total training epochs. Call `step()` once per epoch. |
| `seed` | `int` | `42` | Block-selection seed over the canonical stream. |
| `dataset_seed` | `int \| None` | `None` | Canonical dataset-stream shuffle seed. `None` defaults to `0` internally. |
| `output_dir` | `str \| Path` | `"runs/default"` | Directory for `metrics.jsonl` and checkpoint files. |
| `cache_dir` | `str \| Path \| None` | `None` | Block cache root. `None` = `~/.cache/cr-train`. |

### `Trainer.step() -> dict`

Runs one training epoch + validation + checkpoint. Returns:

```python
{
    "epoch": 1,
    "train": {
        "loss": 0.0423,
        "metrics": {"mae": 0.0312},
        "num_samples": 2048,
        "num_batches": 512,
        "samples_per_sec": 142.3,
        "batches_per_sec": 17.8,
    },
    "val": {
        "loss": 0.0391,
        "metrics": {"mae": 0.0298},
        "num_samples": 256,
        "num_batches": 64,
    },
    "checkpoint_path": "runs/sen12mscr/epoch-0001.pt",
}
```

### `Trainer.test() -> dict`

Runs test evaluation with the current model state. Returns:

```python
{
    "epoch": 2,
    "loss": 0.0387,
    "metrics": {"mae": 0.0295},
    "num_samples": 256,
    "num_batches": 64,
}
```

---

## How It Works

### Block-based caching

Data from HuggingFace is streamed and partitioned into **16-row blocks** (`BLOCK_SIZE=16`). Each block is stored as an Arrow dataset chunk on disk. The cache is keyed on `split + dataset_seed + shuffle_buffer_size`, so different seed combinations produce independent caches.

```
~/.cache/cr-train/layout-v7/<source>/block_store/<split>/dataset-seed=N-shuffle-buffer=128/
├── chunks/          # Arrow dataset chunks, one per block
├── state.json       # Cache state (frontier, seed info)
├── chunk_ids.npy    # Block → chunk mapping
└── cached.npy       # Boolean mask of cached blocks
```

### Two-seed deterministic sampling

The system uses two independent seeds for full reproducibility:

| Seed | Controls | Effect |
|------|----------|--------|
| `dataset_seed` | Canonical shuffled stream | `dataset.shuffle(seed=dataset_seed, buffer_size=128)` fixes a deterministic row ordering per split. |
| `seed` | Block selection | A sequential additive planner selects exactly `ceil(requested_rows / 16)` blocks from a candidate window of `2 * required_blocks`. |

The take probability increases as the candidate window shrinks, guaranteeing the exact block count. Same `seed` = same block selection across runs. Different `seed` values sample different blocks from the same canonical stream.

### Cache warmup lifecycle

1. On the first `step()` or `test()`, warmup runs for all three splits (train, validation, test).
2. A `CachePlan` compares selected blocks against already-cached blocks.
3. Only missing blocks are fetched from HuggingFace. If every block is already cached, HuggingFace is never contacted.
4. A tqdm progress bar tracks block download. On completion, a block timeline is printed:

```
██░░██████░░░░██████████░░██ cache train | warm | hit=42 miss=6 runs=8
```

`█` = selected block, `░` = skipped block in the candidate window.

---

## Distributed Training

Trainer auto-wraps the model in `DistributedDataParallel` when `torch.distributed` is initialized. No code changes needed:

```bash
torchrun --nproc_per_node=2 examples/train_sen12mscr.py \
  --max-train-samples 4096 \
  --epochs 5
```

- Data is sharded across ranks via `DistributedSampler`
- Metrics are all-reduced across all processes
- Only rank 0 writes `metrics.jsonl` and checkpoint files
- Cache warmup runs on all ranks with file-lock coordination

---

## Model Contract

Your model's `forward` method receives **two positional arguments**:

| Argument | Shape | Dtype | Description |
|----------|-------|-------|-------------|
| `sar` | `[B, 2, 256, 256]` | `float32` | Sentinel-1 SAR image (2 channels) |
| `cloudy` | `[B, 13, 256, 256]` | `float32` | Cloudy Sentinel-2 optical image (13 channels) |

**Output**: a prediction tensor, typically `[B, 13, 256, 256]`.

```python
class MyModel(nn.Module):
    def forward(self, sar, cloudy):
        # sar:    [B, 2,  256, 256]
        # cloudy: [B, 13, 256, 256]
        x = torch.cat([sar, cloudy], dim=1)  # [B, 15, 256, 256]
        return self.network(x)
```

The **loss** and **metric** functions receive `(prediction, batch)` where `batch` is the full dict containing `"sar"`, `"cloudy"`, `"target"`, and `"meta"`:

```python
def my_loss(prediction, batch):
    return F.l1_loss(prediction, batch["target"])
```

---

## Notes

- Cache warmup shows a tqdm progress bar during block download and prints a block timeline on completion.
- Equal `seed` values keep the same block-selection membership; train batch order still changes by epoch via `seed + epoch_index`.
- Finished caches are never auto-deleted. Remove them manually from the cache directory to reclaim disk space.
- `Trainer.step()` shows running-average loss and metrics with batch-level tqdm during training.
- Checkpoints are saved as `epoch-NNNN.pt` containing `model`, `optimizer`, `epoch`, and `global_step` state dicts.
- Metrics are appended to `metrics.jsonl` in the output directory (one JSON object per line).

---

## License

This project is currently unlicensed. Please add a `LICENSE` file to specify your terms.
