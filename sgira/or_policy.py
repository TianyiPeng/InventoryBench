"""Transparent OR base-stock baseline used by every SGIRA route."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Sequence

from scipy.stats import norm

from .environment import InventoryObservation


@dataclass(frozen=True)
class ORPolicyConfig:
    """Frozen parameters for the OR executor."""

    order_cap: int | None = None
    history_window: int | None = None
    cap_quantile: float | None = None

    def __post_init__(self) -> None:
        if self.order_cap is not None and self.order_cap < 0:
            raise ValueError("order_cap must be non-negative")
        if self.history_window is not None and self.history_window < 1:
            raise ValueError("history_window must be positive")
        if self.cap_quantile is not None and not 0 < self.cap_quantile < 1:
            raise ValueError("cap_quantile must be strictly between 0 and 1")


@dataclass(frozen=True)
class ORDecision:
    route: str
    demand_mean: float
    demand_std: float
    protection_horizon: float
    protection_demand_mean: float
    protection_demand_std: float
    target_inventory: float
    inventory_position: float
    uncapped_order_quantity: int
    order_quantity: int
    computed_order_cap: int | None = None


class ORPolicy:
    """Compute the document's capped base-stock decision deterministically."""

    def __init__(self, config: ORPolicyConfig | None = None) -> None:
        self.config = config or ORPolicyConfig()

    def decide(self, observation: InventoryObservation) -> ORDecision:
        history: Sequence[float] = observation.demand_history
        if self.config.history_window is not None:
            history = history[-self.config.history_window :]

        demand_mean = mean(history) if history else 0.0
        demand_std = stdev(history) if len(history) > 1 else 0.0
        horizon = observation.expected_lead_time + 1.0
        protection_mean = demand_mean * horizon
        protection_std = demand_std * math.sqrt(horizon)
        z_value = float(norm.ppf(observation.critical_fractile))
        target = max(protection_mean + z_value * protection_std, 0.0)
        inventory_position = float(observation.inventory_position)
        uncapped = max(int(math.ceil(target - inventory_position)), 0)
        order = uncapped
        computed_cap: int | None = None
        if self.config.cap_quantile is not None:
            cap_value = demand_mean + float(norm.ppf(self.config.cap_quantile)) * demand_std
            computed_cap = max(int(math.ceil(cap_value)), 0)
            order = min(order, computed_cap)
        if self.config.order_cap is not None:
            order = min(order, self.config.order_cap)

        return ORDecision(
            route="OR",
            demand_mean=float(demand_mean),
            demand_std=float(demand_std),
            protection_horizon=horizon,
            protection_demand_mean=float(protection_mean),
            protection_demand_std=float(protection_std),
            target_inventory=float(target),
            inventory_position=inventory_position,
            uncapped_order_quantity=uncapped,
            order_quantity=order,
            computed_order_cap=computed_cap,
        )
