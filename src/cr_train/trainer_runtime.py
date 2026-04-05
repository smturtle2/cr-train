from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from .progress import set_progress_postfix_str


@dataclass(slots=True)
class MetricAccumulator:
    """배치별 메트릭의 가중 합 누적기. averages()로 샘플 가중 평균 산출."""

    weighted_sums: dict[str, float] = field(default_factory=dict)
    total_examples: int = 0
    total_batches: int = 0

    def update(self, values: Mapping[str, float], batch_size: int) -> None:
        self.total_examples += batch_size
        self.total_batches += 1
        for key, value in values.items():
            self.weighted_sums[key] = self.weighted_sums.get(key, 0.0) + (value * batch_size)

    def averages(self) -> dict[str, float]:
        if self.total_examples == 0:
            return {}
        return {key: value / self.total_examples for key, value in self.weighted_sums.items()}


def compute_loss(loss_fn: Callable[[Any, Mapping[str, Any]], torch.Tensor | float | int], model_output: Any, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    """loss_fn 실행 결과를 텐서로 변환. float/int 반환도 허용."""
    loss_value = loss_fn(model_output, batch)
    if not isinstance(loss_value, torch.Tensor):
        return torch.as_tensor(loss_value, dtype=torch.float32, device=device)
    return loss_value


def _to_float(value: Any, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must return a scalar tensor")
        return float(value.item())
    if isinstance(value, (float, int)):
        return float(value)
    raise TypeError(f"{name} must return a scalar tensor, float, or int")


def compute_metric_values(metric_fns: Mapping[str, Callable[[Any, Mapping[str, Any]], Any]], model_output: Any, batch: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: _to_float(metric_fn(model_output, batch), name)
        for name, metric_fn in metric_fns.items()
    }


def _format_metric(value: float) -> str:
    """메트릭 값을 크기에 맞게 포맷. 작은 값은 자릿수를 늘려 정보 손실을 방지."""
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value < 0.0001:
        return f"{value:.2e}"
    if abs_value < 0.01:
        return f"{value:.5f}"
    if abs_value < 10:
        return f"{value:.4f}"
    return f"{value:.2f}"


def _reduce_progress_state(
    *,
    accumulator: MetricAccumulator,
    reduce_int: Callable[[int], int],
    reduce_sum: Callable[[float], float],
    distributed: bool,
) -> tuple[int, int, dict[str, float]]:
    reduced_examples = accumulator.total_examples
    reduced_batches = accumulator.total_batches
    reduced_sums = dict(accumulator.weighted_sums)
    if distributed:
        reduced_examples = reduce_int(reduced_examples)
        reduced_batches = reduce_int(reduced_batches)
        reduced_sums = {key: reduce_sum(value) for key, value in reduced_sums.items()}
    return reduced_examples, reduced_batches, reduced_sums


def update_progress_bar(
    progress: Any,
    *,
    accumulator: MetricAccumulator,
    start_time: float | None,
    reduce_int: Callable[[int], int],
    reduce_sum: Callable[[float], float],
    distributed: bool,
) -> None:
    reduced_examples, reduced_batches, reduced_sums = _reduce_progress_state(
        accumulator=accumulator,
        reduce_int=reduce_int,
        reduce_sum=reduce_sum,
        distributed=distributed,
    )
    del start_time, reduced_batches
    if getattr(progress, "disable", False):
        return
    progress.update(1)
    postfix_parts = [
        f"{key}: {_format_metric(value / reduced_examples)}"
        for key, value in reduced_sums.items()
        if reduced_examples > 0
    ]
    set_progress_postfix_str(progress, ", ".join(postfix_parts))


def finalize_summary(
    *,
    accumulator: MetricAccumulator,
    start_time: float | None,
    include_speed: bool,
    reduce_int: Callable[[int], int],
    reduce_sum: Callable[[float], float],
    distributed: bool,
) -> dict[str, Any]:
    reduced_examples, reduced_batches, reduced_sums = _reduce_progress_state(
        accumulator=accumulator,
        reduce_int=reduce_int,
        reduce_sum=reduce_sum,
        distributed=distributed,
    )

    averages = {}
    if reduced_examples > 0:
        averages = {key: value / reduced_examples for key, value in reduced_sums.items()}

    summary = {
        "loss": averages.pop("loss", 0.0),
        "metrics": averages,
        "num_samples": reduced_examples,
        "num_batches": reduced_batches,
    }
    if include_speed:
        elapsed = max((time.perf_counter() - start_time) if start_time is not None else 0.0, 1e-9)
        summary["samples_per_sec"] = reduced_examples / elapsed if reduced_examples > 0 else 0.0
        summary["batches_per_sec"] = reduced_batches / elapsed if reduced_batches > 0 else 0.0
    return summary


def prime_iterator(loader):
    iterator = iter(loader)
    try:
        first_batch = next(iterator)
    except StopIteration:
        return None
    return first_batch, iterator
