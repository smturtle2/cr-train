from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch


def test_training_example_loads_without_running_main() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "train_sen12mscr.py"
    namespace = runpy.run_path(str(example_path), run_name="example_module")

    assert "FusionBaseline" in namespace
    assert "WarmupCosineScheduler" in namespace
    assert "build_scheduler" in namespace
    assert "main" in namespace
    assert namespace["parse_max_samples"]("none") is None
    assert namespace["parse_max_samples"]("full") is None
    assert namespace["parse_max_samples"]("128") == 128
    assert namespace["parse_non_negative_int"]("0") == 0
    assert namespace["parse_positive_int"]("4") == 4


def test_training_example_parser_accepts_train_augmentation_flags(monkeypatch) -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "train_sen12mscr.py"
    namespace = runpy.run_path(str(example_path), run_name="example_module")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sen12mscr.py",
            "--train-crop-size",
            "128",
            "--accum-steps",
            "4",
            "--train-random-flip",
            "--train-random-rot90",
            "--scheduler",
            "warmup-cosine",
            "--scheduler-timing",
            "before_optimizer_step",
            "--warmup-epochs",
            "2",
            "--min-lr-scale",
            "0.2",
        ],
    )
    args = namespace["parse_args"]()

    assert args.train_crop_size == 128
    assert args.accum_steps == 4
    assert args.train_random_flip is True
    assert args.train_random_rot90 is True
    assert args.scheduler == "warmup-cosine"
    assert args.scheduler_timing == "before_optimizer_step"
    assert args.warmup_epochs == 2
    assert args.min_lr_scale == 0.2


def test_training_example_builds_custom_scheduler() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "train_sen12mscr.py"
    namespace = runpy.run_path(str(example_path), run_name="example_module")
    model = namespace["FusionBaseline"]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = namespace["build_scheduler"](
        optimizer,
        scheduler_name="warmup-cosine",
        epochs=4,
        warmup_epochs=1,
        min_lr_scale=0.2,
    )

    assert isinstance(scheduler, namespace["WarmupCosineScheduler"])
    assert namespace["build_scheduler"](
        optimizer,
        scheduler_name="none",
        epochs=4,
        warmup_epochs=1,
        min_lr_scale=0.2,
    ) is None


def test_training_example_parser_defaults_scheduler_timing_to_after_validation(
    monkeypatch,
) -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "train_sen12mscr.py"
    namespace = runpy.run_path(str(example_path), run_name="example_module")

    monkeypatch.setattr(sys, "argv", ["train_sen12mscr.py"])
    args = namespace["parse_args"]()

    assert args.scheduler_timing == "after_validation"


def test_training_example_main_forwards_scheduler_timing(monkeypatch, tmp_path: Path) -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "train_sen12mscr.py"
    namespace = runpy.run_path(str(example_path), run_name="example_module")
    trainer_kwargs: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, model, optimizer, loss, **kwargs) -> None:
            del model, optimizer, loss
            trainer_kwargs.update(kwargs)

        def step(self) -> dict[str, object]:
            return {}

        def test(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_sen12mscr.py",
            "--epochs",
            "1",
            "--output-dir",
            str(tmp_path / "run"),
            "--scheduler-timing",
            "after_optimizer_step",
        ],
    )
    main = namespace["main"]
    main.__globals__["Trainer"] = FakeTrainer

    main()

    assert trainer_kwargs["scheduler_timing"] == "after_optimizer_step"


def test_bitmask_demo_example_loads_without_running_main() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "bitmask_sampling_demo.py"
    namespace = runpy.run_path(str(example_path), run_name="example_module")

    assert "main" in namespace
    assert "build_selection_trace" in namespace
    result = namespace["build_selection_trace"](total_rows=128, requested_rows=48, seed=7)

    assert "main" in namespace
    assert result["required_blocks"] == 1
    assert result["total_blocks"] == 2
    assert result["planner_mode"] == "uniform_exact_k"
    assert len(result["draw_order"]) == 1
    assert len(result["selected_blocks"]) == 1
