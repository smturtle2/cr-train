from __future__ import annotations

import hashlib
import io
import importlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from cr_train.data import (
    BLOCK_SIZE,
    CRPACK_BLOCK_SIZE,
    PreparedSplit,
    build_collate_fn,
    build_dataloader,
    decode_row,
    plan_sample,
    trace_plan_sample,
)
from cr_train.data.dataset import (
    BlockIterableDataset,
    prepare_split,
    prepare_split_from_state,
    resolve_prepared_split_state,
)
from cr_train.data.v15 import pack_v15_block


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _make_row(index: int) -> dict[str, object]:
    sar = (np.arange(256 * 256 * 2, dtype=np.float32) + index).reshape(256, 256, 2)
    cloudy = (np.arange(256 * 256 * 13, dtype=np.int16) + index).reshape(256, 256, 13)
    target = (np.arange(256 * 256 * 13, dtype=np.int16) + index + 10).reshape(256, 256, 13)
    return {
        "sar": sar.tobytes(),
        "cloudy": cloudy.tobytes(),
        "target": target.tobytes(),
        "sar_shape": [256, 256, 2],
        "opt_shape": [256, 256, 13],
        "season": "spring",
        "scene": str(index),
        "patch": f"p{index:03d}",
    }


def _make_non_finite_sar_row(index: int) -> dict[str, object]:
    row = _make_row(index)
    sar = np.empty((256, 256, 2), dtype=np.float32)
    sar[..., 0] = -10.0
    sar[..., 1] = -20.0
    sar[0, 0, 0] = np.nan
    sar[100, 100, 1] = np.nan
    row["sar"] = sar.tobytes()
    return row


def _make_crop_sensitive_sar_row(index: int) -> dict[str, object]:
    row = _make_row(index)
    sar = np.zeros((256, 256, 2), dtype=np.float32)
    sar[64:192, 64:192, 0] = -10.0
    sar[64:192, 64:192, 1] = -20.0
    sar[64, 64, 0] = np.nan
    sar[64, 65, 1] = np.nan
    row["sar"] = sar.tobytes()
    return row


def _make_non_finite_optical_row(index: int) -> dict[str, object]:
    row = _make_row(index)
    cloudy = np.full((256, 256, 13), 1000.0, dtype=np.float32)
    target = np.full((256, 256, 13), 4000.0, dtype=np.float32)
    cloudy[0, 0, 0] = np.nan
    target[10, 11, 3] = np.nan
    row["cloudy"] = cloudy
    row["target"] = target
    return row


class _RowDataset(torch.utils.data.Dataset[dict[str, object]]):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = tuple(dict(row) for row in rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return dict(self.rows[index])


def _selection_seed(seed: int, *, split: str) -> int:
    digest = hashlib.sha256(f"selection:{split}".encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:8], "big")


def _uniform_selected_blocks(seed: int, *, split: str, required_blocks: int, total_blocks: int) -> np.ndarray:
    if required_blocks <= 0 or total_blocks <= 0:
        return np.empty((0,), dtype=np.int64)
    if required_blocks >= total_blocks:
        return np.arange(total_blocks, dtype=np.int64)
    rng = np.random.default_rng(_selection_seed(seed, split=split))
    return np.sort(rng.choice(total_blocks, size=required_blocks, replace=False).astype(np.int64))


def _make_block_splits(block_count: int) -> list[list[dict[str, object]]]:
    blocks: list[list[dict[str, object]]] = []
    current_index = 0
    for _ in range(block_count):
        block_rows = [_make_row(current_index + offset) for offset in range(BLOCK_SIZE)]
        blocks.append(block_rows)
        current_index += BLOCK_SIZE
    return blocks


def _make_scene_block_splits(scene_names: list[str]) -> list[list[dict[str, object]]]:
    blocks: list[list[dict[str, object]]] = []
    current_index = 0
    for scene_name in scene_names:
        block_rows: list[dict[str, object]] = []
        for offset in range(BLOCK_SIZE):
            row = _make_row(current_index + offset)
            row["scene"] = scene_name
            row["patch"] = f"{scene_name}-p{offset:03d}"
            block_rows.append(row)
        blocks.append(block_rows)
        current_index += BLOCK_SIZE
    return blocks


def _catalog(split: str, blocks: list[list[dict[str, object]]]) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    rows_by_key: dict[str, list[dict[str, object]]] = {}
    block_entries = []
    block_row_counts = []
    total_rows = 0
    for source_index, rows in enumerate(blocks):
        for row_start in range(0, len(rows), CRPACK_BLOCK_SIZE):
            row_count = min(CRPACK_BLOCK_SIZE, len(rows) - row_start)
            cache_key = hashlib.sha256(
                f"{split}:{source_index}:{row_start}:{row_count}".encode("utf-8")
            ).hexdigest()[:16]
            rows_by_key[cache_key] = [dict(row) for row in rows[row_start : row_start + row_count]]
            block_entries.append(
                {
                    "index": len(block_entries),
                    "block_id": cache_key,
                    "cache_key": cache_key,
                    "path": f"{split}/unit/scene_{source_index}/block-{len(block_entries):05d}.crpack",
                    "row_start": row_start,
                    "row_count": row_count,
                }
            )
            block_row_counts.append(row_count)
        total_rows += len(rows)
    return {
        "cache_layout_version": 15,
        "cache_block_size": CRPACK_BLOCK_SIZE,
        "split": split,
        "total_rows": total_rows,
        "total_blocks": len(block_entries),
        "block_row_counts": block_row_counts,
        "blocks": block_entries,
    }, rows_by_key


def _make_small_row(index: int) -> dict[str, object]:
    sar = (np.arange(4 * 4 * 2, dtype=np.float32) + index).reshape(4, 4, 2)
    cloudy = (np.arange(4 * 4 * 13, dtype=np.int16) + index).reshape(4, 4, 13)
    target = (np.arange(4 * 4 * 13, dtype=np.int16) + index + 10).reshape(4, 4, 13)
    return {
        "sar": sar.tobytes(),
        "cloudy": cloudy.tobytes(),
        "target": target.tobytes(),
        "sar_shape": [4, 4, 2],
        "opt_shape": [4, 4, 13],
        "season": "winter",
        "scene": f"small-scene-{index}",
        "patch": f"small-p{index:03d}",
    }


def _patch_split_data(monkeypatch, tmp_path: Path, split_blocks: dict[str, list[list[dict[str, object]]]]) -> dict[str, object]:
    del tmp_path
    catalogs: dict[str, dict[str, object]] = {}
    rows_by_key: dict[str, list[dict[str, object]]] = {}
    split_sizes: dict[str, int] = {}
    for split, blocks in split_blocks.items():
        catalog, block_rows = _catalog(split, blocks)
        catalogs[split] = catalog
        rows_by_key.update(block_rows)
        split_sizes[split] = int(catalog["total_rows"])

    load_counts: dict[str, int] = defaultdict(int)
    rows_by_path = {
        str(block["path"]): rows_by_key[str(block["cache_key"])]
        for catalog in catalogs.values()
        for block in catalog["blocks"]
    }

    def fake_load_hf_v2_manifest(*, dataset_root=None, streaming=True):
        del dataset_root, streaming
        return {
            "layout": "cr-hf-scene-v1",
            "block_format_version": 15,
            "splits": {
                split: {
                    "row_count": split_sizes[split],
                    "block_count": int(catalogs[split]["total_blocks"]),
                    "path": f"catalogs/{split}.json",
                }
                for split in split_sizes
            },
        }

    def fake_load_hf_v2_split_catalog(*, split: str, dataset_root=None, streaming=True):
        del dataset_root, streaming
        return catalogs[split]

    def fake_download_remote_file(relative_path: str, target: Path) -> int:
        rows = [dict(row) for row in rows_by_path[relative_path]]
        cache_key = next(
            str(block["cache_key"])
            for catalog in catalogs.values()
            for block in catalog["blocks"]
            if str(block["path"]) == relative_path
        )
        load_counts[cache_key] += 1
        payload_metadata = {
            "row_count": len(rows),
            "season": [str(row["season"]) for row in rows],
            "scene": [str(row["scene"]) for row in rows],
            "patch": [str(row["patch"]) for row in rows],
            "sar_shape": [list(row["sar_shape"]) for row in rows],
            "opt_shape": [list(row["opt_shape"]) for row in rows],
        }
        sar = np.stack(
            [
                np.frombuffer(row["sar"], dtype=np.float32).reshape(row["sar_shape"])
                for row in rows
            ]
        )
        cloudy = np.stack(
            [
                np.frombuffer(row["cloudy"], dtype=np.int16).reshape(row["opt_shape"])
                for row in rows
            ]
        )
        target_array = np.stack(
            [
                np.frombuffer(row["target"], dtype=np.int16).reshape(row["opt_shape"])
                for row in rows
            ]
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            pack_v15_block(
                payload_metadata=payload_metadata,
                metadata={"cache_key": cache_key, "row_count": len(rows)},
                sar=sar,
                cloudy=cloudy,
                target=target_array,
            )
        )
        return target.stat().st_size

    monkeypatch.setattr("cr_train.data.dataset.load_hf_v2_manifest", fake_load_hf_v2_manifest)
    monkeypatch.setattr("cr_train.data.dataset.load_hf_v2_split_catalog", fake_load_hf_v2_split_catalog)
    monkeypatch.setattr("cr_train.data.hf_v2._download_remote_file", fake_download_remote_file)
    return {
        "catalogs": catalogs,
        "rows_by_key": rows_by_key,
        "load_counts": load_counts,
    }


def test_data_package_public_surface_is_minimal_and_explicit() -> None:
    data_mod = importlib.import_module("cr_train.data")

    assert set(data_mod.__all__) == {
        "BLOCK_SIZE",
        "CRPACK_BLOCK_SIZE",
        "BlockIterableDataset",
        "DATASET_ID",
        "OPTICAL_CHANNELS",
        "PreparedSplit",
        "SAR_CHANNELS",
        "SamplePlan",
        "SelectionTrace",
        "build_collate_fn",
        "build_dataloader",
        "decode_row",
        "move_batch_to_device",
        "plan_sample",
        "resolve_num_workers",
        "seed_everything",
        "seed_worker",
        "trace_plan_sample",
    }
    assert all(hasattr(data_mod, name) for name in data_mod.__all__)


def test_decode_row_converts_to_chw_and_normalizes() -> None:
    decoded = decode_row(_make_row(3))

    assert decoded["sar"].shape == (2, 256, 256)
    assert decoded["cloudy"].shape == (13, 256, 256)
    assert decoded["target"].shape == (13, 256, 256)
    assert decoded["sar"].dtype == np.float32
    assert decoded["cloudy"].dtype == np.float32
    assert decoded["target"].dtype == np.float32
    assert decoded["meta"]["scene"] == "3"


def test_decode_row_fills_sar_nan_with_image_mean_before_normalization() -> None:
    row = _make_non_finite_sar_row(5)
    decoded = decode_row(row)

    assert bool(np.isfinite(decoded["sar"]).all())
    assert decoded["sar"][0, 0, 0] == pytest.approx(0.8)
    assert decoded["sar"][1, 100, 100] == pytest.approx(17.5 * 2.0 / 32.5)
    assert decoded["sar"][0, 0, 1] == pytest.approx(1.2)
    assert decoded["sar"][1, 100, 101] == pytest.approx(12.5 * 2.0 / 32.5)


def test_decode_row_fills_optical_nan_with_image_mean_before_normalization() -> None:
    row = _make_non_finite_optical_row(6)
    decoded = decode_row(row)

    assert bool(np.isfinite(decoded["cloudy"]).all())
    assert bool(np.isfinite(decoded["target"]).all())
    assert decoded["cloudy"][0, 0, 0] == pytest.approx(0.5)
    assert decoded["target"][3, 10, 11] == pytest.approx(2.0)


def test_build_collate_fn_batches_rows() -> None:
    collate = build_collate_fn()
    batch = collate([_make_row(0), _make_row(1)])

    assert batch["sar"].shape == (2, 2, 256, 256)
    assert batch["cloudy"].shape == (2, 13, 256, 256)
    assert batch["target"].shape == (2, 13, 256, 256)
    assert batch["meta"]["patch"] == ["p000", "p001"]


def test_build_collate_fn_fills_sar_nan_with_image_mean_without_spatial_transform() -> None:
    collate = build_collate_fn()
    row = _make_non_finite_sar_row(7)
    batch = collate([row])

    assert bool(torch.isfinite(batch["sar"]).all())
    assert batch["sar"][0, 0, 0, 0].item() == pytest.approx(0.8)
    assert batch["sar"][0, 1, 100, 100].item() == pytest.approx(17.5 * 2.0 / 32.5)
    assert batch["sar"][0, 0, 0, 1].item() == pytest.approx(1.2)
    assert batch["sar"][0, 1, 100, 101].item() == pytest.approx(12.5 * 2.0 / 32.5)


def test_build_collate_fn_uses_full_image_mean_for_nan_fill_before_crop() -> None:
    row = _make_crop_sensitive_sar_row(8)
    collate = build_collate_fn(crop_size=128, crop_mode="center")

    batch = collate([row])
    decoded = decode_row(row)

    assert bool(torch.isfinite(batch["sar"]).all())
    expected = torch.from_numpy(decoded["sar"][:, 64:192, 64:192].copy())
    torch.testing.assert_close(batch["sar"][0], expected)


def test_build_collate_fn_raises_if_tensor_remains_non_finite_after_normalization(
    monkeypatch,
) -> None:
    import cr_train.data.dataset as dataset_mod

    original_normalize = dataset_mod._normalize_sar_tensor

    def inject_nan(sar: torch.Tensor) -> None:
        original_normalize(sar)
        sar[0, 0, 0] = float("nan")

    monkeypatch.setattr(dataset_mod, "_normalize_sar_tensor", inject_nan)
    collate = build_collate_fn()

    with pytest.raises(FloatingPointError, match="non-finite sar after normalization"):
        collate([_make_row(0)])


def test_build_collate_fn_is_picklable() -> None:
    collate = build_collate_fn(crop_size=128, crop_mode="center")

    restored = pickle.loads(pickle.dumps(collate))
    batch = restored([_make_row(0)])

    assert batch["cloudy"].shape == (1, 13, 128, 128)


def test_build_collate_fn_applies_spatial_transforms_consistently(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    randint_values = iter((5, 7))
    random_values = iter((0.4, 0.6))
    monkeypatch.setattr(dataset_mod.random, "randint", lambda _lo, _hi: next(randint_values))
    monkeypatch.setattr(dataset_mod.random, "random", lambda: next(random_values))
    monkeypatch.setattr(dataset_mod.random, "randrange", lambda _stop: 1)

    row = _make_row(0)
    collate = build_collate_fn(
        crop_size=128,
        crop_mode="random",
        random_flip=True,
        random_rot90=True,
    )
    batch = collate([row])
    decoded = decode_row(row)

    expected_sar = torch.from_numpy(decoded["sar"][:, 5:133, 7:135].copy())
    expected_sar = torch.flip(expected_sar, dims=(-2,))
    expected_sar = torch.rot90(expected_sar, k=1, dims=(-2, -1))

    expected_cloudy = torch.from_numpy(decoded["cloudy"][:, 5:133, 7:135].copy())
    expected_cloudy = torch.flip(expected_cloudy, dims=(-2,))
    expected_cloudy = torch.rot90(expected_cloudy, k=1, dims=(-2, -1))

    expected_target = torch.from_numpy(decoded["target"][:, 5:133, 7:135].copy())
    expected_target = torch.flip(expected_target, dims=(-2,))
    expected_target = torch.rot90(expected_target, k=1, dims=(-2, -1))

    assert batch["sar"].shape == (1, 2, 128, 128)
    assert batch["cloudy"].shape == (1, 13, 128, 128)
    assert batch["target"].shape == (1, 13, 128, 128)
    torch.testing.assert_close(batch["sar"][0], expected_sar)
    torch.testing.assert_close(batch["cloudy"][0], expected_cloudy)
    torch.testing.assert_close(batch["target"][0], expected_target)


def test_build_collate_fn_center_crop_reduces_spatial_size() -> None:
    row = _make_row(1)
    collate = build_collate_fn(crop_size=128, crop_mode="center")
    batch = collate([row])
    decoded = decode_row(row)

    expected = torch.from_numpy(decoded["cloudy"][:, 64:192, 64:192].copy())

    assert batch["cloudy"].shape == (1, 13, 128, 128)
    torch.testing.assert_close(batch["cloudy"][0], expected)


def test_build_collate_fn_rejects_oversized_crop() -> None:
    collate = build_collate_fn(crop_size=512, crop_mode="random")

    with pytest.raises(ValueError, match="crop_size"):
        collate([_make_row(0)])


def test_build_collate_fn_requires_crop_size_for_random_or_center_crop() -> None:
    with pytest.raises(ValueError, match="crop_size"):
        build_collate_fn(crop_mode="random")

    with pytest.raises(ValueError, match="crop_size"):
        build_collate_fn(crop_mode="center")


def test_plan_sample_is_block_reproducible_within_total_block_domain() -> None:
    catalog = {"total_rows": 10 * BLOCK_SIZE, "total_blocks": 10}

    requested_rows = (2 * BLOCK_SIZE) - 1
    sample_a = plan_sample(catalog, seed=7, max_samples=requested_rows)
    sample_b = plan_sample(catalog, seed=7, max_samples=requested_rows)
    distinct_plans = {
        tuple(plan_sample(catalog, seed=seed, max_samples=3 * BLOCK_SIZE).selected_blocks.tolist())
        for seed in range(64)
    }

    assert sample_a.requested_rows == requested_rows
    assert sample_a.required_blocks == 2
    assert sample_a.effective_rows == 2 * BLOCK_SIZE
    assert sample_a.total_blocks == 10
    assert sample_a.planner_mode == "uniform_exact_k"
    assert sample_a.selected_blocks.size == sample_a.required_blocks
    assert np.array_equal(sample_a.selected_blocks, sample_b.selected_blocks)
    assert len(distinct_plans) > 1
    assert np.all(sample_a.selected_blocks < sample_a.total_blocks)
    assert sample_a.execution_block_count == int(sample_a.selected_blocks[-1]) + 1


def test_plan_sample_full_request_selects_all_blocks_for_full_split() -> None:
    catalog = {"total_rows": 10, "total_blocks": 10}

    implicit_full = plan_sample(catalog, seed=7, max_samples=None, split="train")
    explicit_full = plan_sample(catalog, seed=7, max_samples=999, split="train")

    for sample in (implicit_full, explicit_full):
        assert sample.requested_rows == 10
        assert sample.effective_rows == 10
        assert sample.required_blocks == 10
        assert sample.total_blocks == 10
        assert sample.planner_mode == "full_split"
        assert sample.selected_blocks.tolist() == list(range(10))
        assert np.all(sample.selected_bitmap)
        assert sample.execution_block_count == 10


def test_trace_plan_sample_reports_uniform_exact_k_metadata() -> None:
    trace = trace_plan_sample({"total_rows": 10 * BLOCK_SIZE}, seed=11, max_samples=3 * BLOCK_SIZE, split="train")

    assert trace.total_blocks == 10
    assert trace.requested_rows == 3 * BLOCK_SIZE
    assert trace.required_blocks == 3
    assert trace.planner_mode == "uniform_exact_k"
    assert trace.selected_blocks.size == 3
    assert trace.draw_order.size == 3
    assert trace.execution_block_count == int(trace.selected_blocks[-1]) + 1


def test_trace_plan_sample_reports_full_split_metadata_for_full_request() -> None:
    trace = trace_plan_sample({"total_rows": 10, "total_blocks": 10}, seed=11, max_samples=None, split="train")

    assert trace.total_blocks == 10
    assert trace.requested_rows == 10
    assert trace.required_blocks == 10
    assert trace.planner_mode == "full_split"
    assert trace.draw_order.tolist() == list(range(10))
    assert trace.selected_blocks.tolist() == list(range(10))
    assert np.all(trace.selected_bitmap)
    assert trace.execution_block_count == 10


def test_resolve_prepared_split_state_uses_catalog_metadata(monkeypatch, tmp_path: Path) -> None:
    split_blocks = {"train": _make_block_splits(2)}
    _patch_split_data(monkeypatch, tmp_path, split_blocks)

    state = resolve_prepared_split_state(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=7,
        dataset_root=tmp_path,
        streaming=False,
    )

    assert state.required_blocks == 4
    assert len(state.row_counts_by_key) == 4
    assert sum(state.row_counts_by_key.values()) == 2 * BLOCK_SIZE
    assert set(state.row_counts_by_key.values()) == {CRPACK_BLOCK_SIZE}


def test_prepare_split_reads_blocks_in_selected_order(monkeypatch, tmp_path: Path) -> None:
    split_blocks = {
        "train": _make_block_splits(4),
        "validation": _make_block_splits(4),
        "test": _make_block_splits(4),
    }
    patched = _patch_split_data(monkeypatch, tmp_path, split_blocks)

    prepared = prepare_split(
        split="validation",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=13,
        epoch=0,
        training=False,
        dataset_root=tmp_path,
        streaming=False,
    )
    loader = build_dataloader(
        prepared,
        batch_size=8,
        num_workers=0,
        training=False,
        seed=13,
        epoch=0,
    )

    batch_scenes = [scene for batch in loader for scene in batch["meta"]["scene"]]
    sample_plan = plan_sample(
        patched["catalogs"]["validation"],
        seed=13,
        max_samples=2 * BLOCK_SIZE,
        split="validation",
    )
    expected_scenes: list[str] = []
    for block_index in sample_plan.selected_blocks.tolist():
        block = patched["catalogs"]["validation"]["blocks"][int(block_index)]
        expected_scenes.extend(row["scene"] for row in patched["rows_by_key"][str(block["cache_key"])])

    assert batch_scenes == expected_scenes


def test_prepare_split_training_order_changes_by_epoch(monkeypatch, tmp_path: Path) -> None:
    split_blocks = {
        "train": _make_block_splits(4),
        "validation": _make_block_splits(2),
        "test": _make_block_splits(2),
    }
    _patch_split_data(monkeypatch, tmp_path, split_blocks)

    state = resolve_prepared_split_state(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=2 * BLOCK_SIZE,
        seed=9,
        dataset_root=tmp_path,
        streaming=False,
    )
    prepared_epoch0 = prepare_split_from_state(
        state,
        epoch=0,
        training=True,
    )
    prepared_epoch1 = prepare_split_from_state(
        state,
        epoch=1,
        training=True,
    )
    loader_epoch0 = build_dataloader(
        prepared_epoch0,
        batch_size=8,
        num_workers=0,
        training=True,
        seed=9,
        epoch=0,
    )
    loader_epoch1 = build_dataloader(
        prepared_epoch1,
        batch_size=8,
        num_workers=0,
        training=True,
        seed=9,
        epoch=1,
    )

    scenes_epoch0 = [scene for batch in loader_epoch0 for scene in batch["meta"]["scene"]]
    scenes_epoch1 = [scene for batch in loader_epoch1 for scene in batch["meta"]["scene"]]

    assert scenes_epoch0 != scenes_epoch1
    assert set(scenes_epoch0) == set(scenes_epoch1)


def test_prepare_split_training_order_is_reproducible_for_same_seed_and_epoch(
    monkeypatch, tmp_path: Path
) -> None:
    split_blocks = {
        "train": _make_scene_block_splits(["scene-134", "scene-055", "scene-200", "scene-201"]),
        "validation": _make_block_splits(2),
        "test": _make_block_splits(2),
    }
    _patch_split_data(monkeypatch, tmp_path, split_blocks)

    state = resolve_prepared_split_state(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=4 * BLOCK_SIZE,
        seed=9,
        dataset_root=tmp_path,
        streaming=False,
    )

    prepared_a = prepare_split_from_state(state, epoch=0, training=True)
    prepared_b = prepare_split_from_state(state, epoch=0, training=True)
    loader_a = build_dataloader(
        prepared_a,
        batch_size=8,
        num_workers=0,
        training=True,
        seed=9,
        epoch=0,
    )
    loader_b = build_dataloader(
        prepared_b,
        batch_size=8,
        num_workers=0,
        training=True,
        seed=9,
        epoch=0,
    )

    scenes_a = [scene for batch in loader_a for scene in batch["meta"]["scene"]]
    scenes_b = [scene for batch in loader_b for scene in batch["meta"]["scene"]]

    assert scenes_a == scenes_b


def test_training_iterator_does_not_block_to_fill_active_pool() -> None:
    class ReadinessBlockReader:
        def __init__(self) -> None:
            self.load_calls: list[str] = []
            self.ready_calls: list[str] = []

        def load_block(self, cache_key: str) -> list[dict[str, str]]:
            self.load_calls.append(cache_key)
            return [{"scene": cache_key}, {"scene": cache_key}]

        def block_is_ready(self, cache_key: str) -> bool:
            self.ready_calls.append(cache_key)
            return False

    reader = ReadinessBlockReader()
    dataset = BlockIterableDataset(
        block_reader=reader,
        blocks=(
            {"cache_key": "block-0"},
            {"cache_key": "block-1"},
        ),
        seed=9,
        epoch=0,
        split="train",
        training=True,
    )

    iterator = dataset._iter_training_rows(
        blocks=[
            {"cache_key": "block-0"},
            {"cache_key": "block-1"},
        ],
        worker_id=0,
    )

    assert next(iterator)["scene"] == "block-0"
    assert reader.load_calls == ["block-0"]

    assert next(iterator)["scene"] == "block-0"
    assert reader.load_calls == ["block-0"]
    assert reader.ready_calls == ["block-1"]


def test_prepare_split_training_mixes_samples_across_scene_local_blocks(
    monkeypatch, tmp_path: Path
) -> None:
    split_blocks = {
        "train": _make_scene_block_splits(["scene-134", "scene-055", "scene-200", "scene-201"]),
        "validation": _make_block_splits(2),
        "test": _make_block_splits(2),
    }
    _patch_split_data(monkeypatch, tmp_path, split_blocks)

    prepared = prepare_split(
        split="train",
        dataset_name="unit/test",
        revision=None,
        max_samples=4 * BLOCK_SIZE,
        seed=9,
        epoch=0,
        training=True,
        dataset_root=tmp_path,
        streaming=False,
    )
    loader = build_dataloader(
        prepared,
        batch_size=8,
        num_workers=0,
        training=True,
        seed=9,
        epoch=0,
    )

    first_scenes: list[str] = []
    for batch in loader:
        first_scenes.extend(batch["meta"]["scene"])
        if len(first_scenes) >= 8:
            break

    assert len(first_scenes) == 8
    assert len(set(first_scenes)) == 4


def test_build_dataloader_defaults_to_non_persistent_workers(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    captured: dict[str, Any] = {}

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dataset_mod, "DataLoader", FakeDataLoader)

    prepared = PreparedSplit(dataset=SimpleNamespace(name="dataset"), num_examples=4)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=2,
        training=False,
        seed=5,
        epoch=0,
    )

    assert loader is not None
    assert captured["dataset"] is prepared.dataset
    assert captured["kwargs"]["persistent_workers"] is False
    assert "multiprocessing_context" not in captured["kwargs"]


def test_build_dataloader_passes_multiprocessing_context(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    captured: dict[str, Any] = {}

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dataset_mod, "DataLoader", FakeDataLoader)

    prepared = PreparedSplit(dataset=SimpleNamespace(name="dataset"), num_examples=4)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=2,
        training=False,
        seed=5,
        epoch=0,
        multiprocessing_context="spawn",
    )

    assert loader is not None
    assert captured["dataset"] is prepared.dataset
    assert captured["kwargs"]["multiprocessing_context"] == "spawn"


def test_build_dataloader_ignores_multiprocessing_context_without_workers(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    captured: dict[str, Any] = {}

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dataset_mod, "DataLoader", FakeDataLoader)

    prepared = PreparedSplit(dataset=SimpleNamespace(name="dataset"), num_examples=4)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=0,
        training=False,
        seed=5,
        epoch=0,
        multiprocessing_context="spawn",
    )

    assert loader is not None
    assert captured["dataset"] is prepared.dataset
    assert "multiprocessing_context" not in captured["kwargs"]


def test_build_dataloader_supports_spawn_workers() -> None:
    prepared = PreparedSplit(dataset=_RowDataset([_make_row(0), _make_row(1)]), num_examples=2)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=1,
        training=False,
        seed=5,
        epoch=0,
        multiprocessing_context="spawn",
    )

    batch = next(iter(loader))

    assert batch["sar"].shape == (2, 2, 256, 256)
    assert batch["cloudy"].shape == (2, 13, 256, 256)


def test_build_dataloader_passes_spatial_transform_options(monkeypatch) -> None:
    import cr_train.data.dataset as dataset_mod

    captured: dict[str, Any] = {}

    def fake_build_collate_fn(**kwargs):
        captured["collate_kwargs"] = kwargs
        return "fake-collate"

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dataset_mod, "build_collate_fn", fake_build_collate_fn)
    monkeypatch.setattr(dataset_mod, "DataLoader", FakeDataLoader)

    prepared = PreparedSplit(dataset=SimpleNamespace(name="dataset"), num_examples=4)
    loader = build_dataloader(
        prepared,
        batch_size=2,
        num_workers=0,
        training=True,
        seed=5,
        epoch=0,
        crop_size=128,
        crop_mode="random",
        random_flip=True,
        random_rot90=True,
    )

    assert loader is not None
    assert captured["dataset"] is prepared.dataset
    assert captured["collate_kwargs"] == {
        "include_metadata": True,
        "crop_size": 128,
        "crop_mode": "random",
        "random_flip": True,
        "random_rot90": True,
    }
    assert captured["kwargs"]["collate_fn"] == "fake-collate"
