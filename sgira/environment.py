"""Leakage-safe single-item lost-sales inventory environment.

The environment owns complete demand and supply trajectories. Policies only
receive :class:`InventoryObservation`, which contains information observable
before the current order is placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OutstandingOrder:
    """Observable portion of an order that has not arrived yet."""

    placed_period: int
    quantity: int
    age: int


@dataclass(frozen=True)
class InventoryObservation:
    """State that may legally be passed to a decision method."""

    period: int
    on_hand: float
    outstanding_orders: tuple[OutstandingOrder, ...]
    demand_history: tuple[float, ...]
    expected_lead_time: float
    profit_per_unit: float
    holding_cost_per_unit: float
    critical_fractile: float
    context: Mapping[str, Any]

    @property
    def in_transit_total(self) -> int:
        return sum(order.quantity for order in self.outstanding_orders)

    @property
    def inventory_position(self) -> float:
        return self.on_hand + self.in_transit_total


@dataclass(frozen=True)
class PeriodResult:
    """Auditable result produced after one period has finished."""

    period: int
    order_quantity: int
    arrivals: int
    demand: float
    sales: float
    lost_sales: float
    ending_inventory: float
    reward: float


@dataclass
class _Order:
    placed_period: int
    quantity: int
    arrival_period: int | None


class InventoryEnv:
    """Replay environment with fixed or stochastic lead times.

    ``actual_lead_times`` is indexed by order period, not by the number of
    non-zero orders. Thus all compared policies face the same supply outcome
    for an order placed in a given period. ``None`` represents a permanently
    lost order. It remains visibly in transit because the policy cannot know
    in advance that it will never arrive.
    """

    def __init__(
        self,
        demands: Sequence[float],
        *,
        initial_inventory: int = 0,
        expected_lead_time: float = 0,
        actual_lead_times: Sequence[int | None] | None = None,
        profit_per_unit: float = 1.0,
        holding_cost_per_unit: float = 0.0,
        critical_fractile: float = 0.5,
        initial_demand_history: Sequence[float] = (),
        contexts: Sequence[Mapping[str, Any]] | None = None,
        order_cap: int | None = None,
    ) -> None:
        if not demands:
            raise ValueError("demands must contain at least one period")
        if any(float(value) < 0 for value in demands):
            raise ValueError("demands must be non-negative")
        if initial_inventory < 0:
            raise ValueError("initial_inventory must be non-negative")
        if expected_lead_time < 0:
            raise ValueError("expected_lead_time must be non-negative")
        if profit_per_unit < 0 or holding_cost_per_unit < 0:
            raise ValueError("profit and holding cost must be non-negative")
        if not 0 < critical_fractile < 1:
            raise ValueError("critical_fractile must be strictly between 0 and 1")
        if order_cap is not None and order_cap < 0:
            raise ValueError("order_cap must be non-negative")

        lead_times = (
            tuple(actual_lead_times)
            if actual_lead_times is not None
            else tuple(int(expected_lead_time) for _ in demands)
        )
        if len(lead_times) != len(demands):
            raise ValueError("actual_lead_times must match the demand horizon")
        if any(value is not None and value < 0 for value in lead_times):
            raise ValueError("actual lead times must be non-negative or None")
        if contexts is not None and len(contexts) != len(demands):
            raise ValueError("contexts must match the demand horizon")

        self._demands = tuple(float(value) for value in demands)
        self._actual_lead_times = lead_times
        self._contexts = tuple(contexts or ({} for _ in demands))
        self._initial_inventory = int(initial_inventory)
        self._initial_history = tuple(float(value) for value in initial_demand_history)
        self.expected_lead_time = float(expected_lead_time)
        self.profit_per_unit = float(profit_per_unit)
        self.holding_cost_per_unit = float(holding_cost_per_unit)
        self.critical_fractile = float(critical_fractile)
        self.order_cap = order_cap
        self.reset()

    @property
    def horizon(self) -> int:
        return len(self._demands)

    @property
    def done(self) -> bool:
        return self._period > self.horizon

    @property
    def results(self) -> tuple[PeriodResult, ...]:
        return tuple(self._results)

    def reset(self) -> InventoryObservation:
        self._period = 1
        self._on_hand = float(self._initial_inventory)
        self._orders: list[_Order] = []
        self._history = list(self._initial_history)
        self._results: list[PeriodResult] = []
        return self.observe()

    def observe(self) -> InventoryObservation:
        if self.done:
            raise RuntimeError("episode has finished")
        outstanding = tuple(
            OutstandingOrder(
                placed_period=order.placed_period,
                quantity=order.quantity,
                age=self._period - order.placed_period,
            )
            for order in self._orders
        )
        return InventoryObservation(
            period=self._period,
            on_hand=self._on_hand,
            outstanding_orders=outstanding,
            demand_history=tuple(self._history),
            expected_lead_time=self.expected_lead_time,
            profit_per_unit=self.profit_per_unit,
            holding_cost_per_unit=self.holding_cost_per_unit,
            critical_fractile=self.critical_fractile,
            context=dict(self._contexts[self._period - 1]),
        )

    def step(self, order_quantity: int) -> tuple[InventoryObservation | None, PeriodResult]:
        if self.done:
            raise RuntimeError("episode has finished")
        if isinstance(order_quantity, bool) or not isinstance(order_quantity, int):
            raise TypeError("order_quantity must be an integer")
        if order_quantity < 0:
            raise ValueError("order_quantity must be non-negative")
        if self.order_cap is not None and order_quantity > self.order_cap:
            raise ValueError(f"order_quantity exceeds order_cap={self.order_cap}")

        period = self._period
        if order_quantity:
            lead_time = self._actual_lead_times[period - 1]
            arrival_period = None if lead_time is None else period + lead_time
            self._orders.append(_Order(period, order_quantity, arrival_period))

        arriving = [order for order in self._orders if order.arrival_period == period]
        arrivals = sum(order.quantity for order in arriving)
        self._orders = [order for order in self._orders if order.arrival_period != period]

        available = self._on_hand + arrivals
        demand = self._demands[period - 1]
        sales = min(available, demand)
        lost_sales = max(demand - available, 0.0)
        ending_inventory = max(available - demand, 0.0)
        reward = self.profit_per_unit * sales - self.holding_cost_per_unit * ending_inventory

        result = PeriodResult(
            period=period,
            order_quantity=order_quantity,
            arrivals=arrivals,
            demand=demand,
            sales=sales,
            lost_sales=lost_sales,
            ending_inventory=ending_inventory,
            reward=reward,
        )
        self._results.append(result)
        self._history.append(demand)
        self._on_hand = ending_inventory
        self._period += 1
        return (None if self.done else self.observe()), result
