"""Frozen human-rule routing baseline from the research design."""

from __future__ import annotations

from dataclasses import dataclass

from .executors import Route
from .features import StateFeatures


@dataclass(frozen=True)
class RuleThresholds:
    trend: float = 0.05
    shift: float = 0.20
    volatile_cv: float = 0.75
    delay_score: float = 1.50

    def __post_init__(self) -> None:
        if min(self.trend, self.shift, self.volatile_cv, self.delay_score) < 0:
            raise ValueError("rule thresholds must be non-negative")


@dataclass(frozen=True)
class RuleRoutingDecision:
    route: Route
    demand_abnormal: bool
    supply_abnormal: bool
    reason_codes: tuple[str, ...]


class RuleRouter:
    def __init__(self, thresholds: RuleThresholds | None = None) -> None:
        self.thresholds = thresholds or RuleThresholds()

    def select(self, features: StateFeatures) -> RuleRoutingDecision:
        demand_codes: list[str] = []
        if features.trend >= self.thresholds.trend:
            demand_codes.append("UPWARD_TREND")
        elif features.trend <= -self.thresholds.trend:
            demand_codes.append("DOWNWARD_TREND")
        if abs(features.shift) >= self.thresholds.shift:
            demand_codes.append("DEMAND_SHIFT")
        if features.cv >= self.thresholds.volatile_cv:
            demand_codes.append("HIGH_VOLATILITY")

        supply_codes: list[str] = []
        if features.delay_score >= self.thresholds.delay_score:
            supply_codes.append("ORDER_DELAYED")
        if features.overdue_order_count:
            supply_codes.append("OVERDUE_ORDERS")

        demand_abnormal = bool(demand_codes)
        supply_abnormal = bool(supply_codes)
        if demand_abnormal and supply_abnormal:
            route = Route.LLM
        elif supply_abnormal:
            route = Route.OR_TO_LLM
        elif demand_abnormal:
            route = Route.LLM_TO_OR
        else:
            route = Route.OR
        return RuleRoutingDecision(
            route, demand_abnormal, supply_abnormal,
            tuple(demand_codes + supply_codes),
        )
