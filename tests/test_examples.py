from __future__ import annotations

import runpy
from pathlib import Path


def test_training_example_loads_without_running_main() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "train_sen12mscr.py"
    namespace = runpy.run_path(str(example_path), run_name="example_module")

    assert "FusionBaseline" in namespace
    assert "main" in namespace
    assert namespace["parse_max_samples"]("none") is None
    assert namespace["parse_max_samples"]("full") is None
    assert namespace["parse_max_samples"]("128") == 128


def test_bitmask_demo_example_loads_without_running_main() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "bitmask_sampling_demo.py"
    namespace = runpy.run_path(str(example_path), run_name="example_module")

    assert "main" in namespace
    assert "build_selection_trace" in namespace
    result = namespace["build_selection_trace"](total_rows=128, requested_rows=48, seed=7)

    assert "main" in namespace
    assert result["required_blocks"] == 3
    assert result["candidate_blocks"] == 6
    assert len(result["selected_blocks"]) == 3
