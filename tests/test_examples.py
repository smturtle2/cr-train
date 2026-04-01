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


def test_benchmark_example_loads_without_running_main() -> None:
    example_path = Path(__file__).resolve().parents[1] / "examples" / "benchmark_take_cache.py"
    namespace = runpy.run_path(str(example_path), run_name="example_module")

    assert "main" in namespace
    assert namespace["parse_csv_ints"]("64,256") == [64, 256]
