"""Observable state features for human and Gemini routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

from .environment import InventoryObservation


@dataclass(frozen=True)
class FeatureConfig:
    recent_window: int = 5
    shift_window: int = 3
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.recent_window < 2:
            raise ValueError("recent_window must be >= 2")
        if self.shift_window < 1 or self.epsilon <= 0:
            raise ValueError("shift_window and epsilon must be positive")


@dataclass(frozen=True)
class StateFeatures:
    recent_demand_mean: float
    recent_demand_std: float
    cv: float
    trend: float
    shift: float
    coverage: float
    delay_score: float
    oldest_order_age: int
    overdue_order_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2
    y_mean = _mean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return sum(
        (index - x_mean) * (value - y_mean) for index, value in enumerate(values)
    ) / denominator


def build_state_features(
    observation: InventoryObservation,
    config: FeatureConfig | None = None,
) -> StateFeatures:
    config = config or FeatureConfig()
    history = [float(value) for value in observation.demand_history]
    recent = history[-config.recent_window :]
    recent_mean = _mean(recent)
    recent_std = _sample_std(recent)
    cv = recent_std / (recent_mean + config.epsilon)
    trend = _linear_slope(recent) / (recent_mean + config.epsilon)

    width = config.shift_window
    if len(history) >= 2 * width:
        previous_mean = _mean(history[-2 * width : -width])
        current_mean = _mean(history[-width:])
        shift = (current_mean - previous_mean) / (previous_mean + config.epsilon)
    else:
        shift = 0.0

    expected_protection_demand = recent_mean * (observation.expected_lead_time + 1)
    coverage = observation.inventory_position / (
        expected_protection_demand + config.epsilon
    )
    ages = [order.age for order in observation.outstanding_orders]
    oldest_age = max(ages, default=0)
    delay_score = oldest_age / (observation.expected_lead_time + config.epsilon)
    overdue_count = sum(
        order.age > observation.expected_lead_time
        for order in observation.outstanding_orders
    )
    return StateFeatures(
        recent_demand_mean=recent_mean,
        recent_demand_std=recent_std,
        cv=cv,
        trend=trend,
        shift=shift,
        coverage=coverage,
        delay_score=delay_score,
        oldest_order_age=oldest_age,
        overdue_order_count=overdue_count,
    )
