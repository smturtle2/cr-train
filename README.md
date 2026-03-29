# cr-train

English | [한국어](./README.ko.md)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](./pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-streaming%20trainer-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/huggingface-datasets-yellow.svg)](https://huggingface.co/datasets/Hermanni/sen12mscr)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

Model-agnostic SEN12MS-CR training utilities for deterministic streaming experiments.

## Clone And Setup

```bash
git clone https://github.com/smturtle2/cr-train.git
cd cr-train
uv sync
```

Colab-style onboarding notebook:

- [`examples/colab_quickstart.ipynb`](/home/smturtle2/projects/cr-train/examples/colab_quickstart.ipynb)

The notebook still uses `uv` for fast environment setup; Colab or Jupyter remains the notebook runtime.

Optional but recommended for higher Hugging Face rate limits:

```bash
export HF_TOKEN=your_token
```

## Recommended Usage

This repository is meant to be cloned and used as a local training module. The typical workflow is:

1. Clone the repository and install the environment with `uv sync`.
2. Stay at the repository root when running scripts so the local `cr_train` package is available through the project environment.
3. Start from the example entrypoint if you want a working baseline:

```bash
uv run python examples/minimal_train.py --epochs 1 --train-max-batches 10 --val-max-batches 2
```

4. For your own experiment script, import from `cr_train`, not `src.cr_train`.
   Typical imports are `build_sen12mscr_loaders`, `Trainer`, `TrainerConfig`, and optional metrics such as `MAE`.
5. Build `(train_loader, val_loader, test_loader)`, construct your model/optimizer/loss, and pass them into `Trainer`.
6. Run training with `for history in trainer.step(): ...` and evaluation with `trainer.test()`.

If you want to keep your own training file inside this repository, a common layout is:

- `examples/` for runnable reference scripts
- `scripts/` for one-off experiments
- repository root or a subdirectory script executed with `uv run python ...`

The reference implementation is [`minimal_train.py`](/home/smturtle2/projects/cr-train/examples/minimal_train.py).

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
