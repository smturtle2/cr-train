from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import cr_train.data as data_mod
from cr_train import build_loaders
from cr_train.data import PARQUET_COLUMNS


def _sample_row(*, season: str = "spring", scene: str = "1", patch: str = "p30") -> dict[str, object]:
    sar = np.array(
        [
            [[-30.0, -5.0], [-12.5, 1.0]],
            [[-25.0, -7.5], [0.0, -20.0]],
        ],
        dtype=np.float32,
    )
    optical = np.arange(13 * 2 * 1, dtype=np.int16).reshape(2, 1, 13) * 800 - 10
    return {
        "sar": sar.tobytes(),
        "cloudy": optical.tobytes(),
        "target": (optical + 100).tobytes(),
        "sar_shape": list(sar.shape),
        "opt_shape": list(optical.shape),
        "season": season,
        "scene": scene,
        "patch": patch,
    }


# ---------------------------------------------------------------------------
# Batch decode / collate
# ---------------------------------------------------------------------------


def test_collate_decodes_raw_parquet_rows() -> None:
    rows = [_sample_row(), _sample_row(scene="2")]
    batch = data_mod._collate_sen12mscr_rows(rows)

    assert "inputs" in batch and "target" in batch
    sar, cloudy = batch["inputs"]
    assert sar.shape[0] == 2
    assert cloudy.shape[0] == 2
    assert batch["target"].shape[0] == 2
    assert batch["metadata"]["scene"] == ["1", "2"]


def test_collate_passes_through_pre_decoded_batch() -> None:
    pre_decoded = [
        {"inputs": (torch.zeros(2, 2, 2), torch.zeros(13, 2, 1)), "target": torch.zeros(13, 2, 1)}
    ]
    batch = data_mod._collate_sen12mscr_rows(pre_decoded)
    assert "inputs" in batch


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        data_mod._collate_sen12mscr_rows([])


def test_sar_preprocessing_clamps_and_normalizes() -> None:
    tensor = torch.tensor([-50.0, -25.0, -12.5, 0.0, 10.0])
    result = data_mod._preprocess_sar(tensor)
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert result[0].item() == pytest.approx(0.0)
    assert result[-1].item() == pytest.approx(1.0)


def test_optical_preprocessing_clamps_and_normalizes() -> None:
    tensor = torch.tensor([-100.0, 0.0, 5000.0, 10000.0, 20000.0])
    result = data_mod._preprocess_optical(tensor)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# build_loaders
# ---------------------------------------------------------------------------


def test_build_loaders_creates_three_dataloaders(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class _FakeDataLoader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            created.append(dict(kwargs))

    class _FakeHFDS:
        def __init__(self):
            self.shuffled = False

        def shuffle(self, **kwargs):
            self.shuffled = True
            return self

        def __iter__(self):
            yield from []

    datasets_created: list[dict] = []

    def fake_load_dataset(*args, **kwargs):
        datasets_created.append(kwargs)
        return _FakeHFDS()

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(data_mod, "get_token", lambda: "tok")
    monkeypatch.setattr(data_mod, "DataLoader", _FakeDataLoader)

    build_loaders(4, seed=42)

    assert len(created) == 3
    assert len(datasets_created) == 3

    splits = [d["split"] for d in datasets_created]
    assert splits == ["train", "validation", "test"]

    for d in datasets_created:
        assert d["streaming"] is True
        assert d["columns"] == list(PARQUET_COLUMNS)
        assert d["token"] == "tok"

    for kw in created:
        assert kw["batch_size"] == 4
        assert kw["num_workers"] == 0
        assert kw["collate_fn"] is data_mod._collate_sen12mscr_rows


def test_build_loaders_shuffles_train_only(monkeypatch: pytest.MonkeyPatch) -> None:
    hf_datasets: list = []

    class _FakeHFDS:
        def __init__(self):
            self.shuffled = False

        def shuffle(self, **kwargs):
            self.shuffled = True
            return self

        def __iter__(self):
            yield from []

    def fake_load_dataset(*args, **kwargs):
        ds = _FakeHFDS()
        hf_datasets.append(ds)
        return ds

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(data_mod, "get_token", lambda: None)
    monkeypatch.setattr(data_mod, "DataLoader", lambda *a, **kw: kw)

    build_loaders(2, num_workers=0)

    assert hf_datasets[0].shuffled is True   # train
    assert hf_datasets[1].shuffled is False   # val
    assert hf_datasets[2].shuffled is False   # test


def test_build_loaders_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        build_loaders(0)


# ---------------------------------------------------------------------------
# Worker seeding
# ---------------------------------------------------------------------------


def test_seed_worker_seeds_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: dict[str, int] = {}
    monkeypatch.setattr(data_mod.torch, "initial_seed", lambda: 123456)
    monkeypatch.setattr(data_mod.random, "seed", lambda s: seeds.update(random=s))
    monkeypatch.setattr(data_mod.np.random, "seed", lambda s: seeds.update(numpy=s))

    data_mod._seed_worker(0)
    assert seeds["random"] == seeds["numpy"] == 123456 % (2**32)


# ---------------------------------------------------------------------------
# Local parquet smoke test
# ---------------------------------------------------------------------------


def test_local_parquet_collate(tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    row = _sample_row()
    table = pa.table(
        {
            "sar": [row["sar"]],
            "cloudy": [row["cloudy"]],
            "target": [row["target"]],
            "sar_shape": [row["sar_shape"]],
            "opt_shape": [row["opt_shape"]],
            "season": [row["season"]],
            "scene": [row["scene"]],
            "patch": [row["patch"]],
        }
    )
    pq.write_table(table, parquet_path)

    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=str(parquet_path), split="train", streaming=True)
    rows = list(ds)
    batch = data_mod._collate_sen12mscr_rows(rows)
    assert tuple(batch["target"].shape) == (1, 13, 2, 1)
    assert batch["metadata"]["season"] == ["spring"]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_minimal_train_cli_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "minimal_train.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "--num-workers" in result.stdout
