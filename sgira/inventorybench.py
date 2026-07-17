"""InventoryBench loading, fixed-route simulation, and M4 metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .decision_logger import DecisionLogger
from .controller import ControllerConfig, GeminiController, MemoryItem, RoutingRecord
from .dynamic_policy import DynamicDecision, RuleRoutingPolicy, SGIRAPolicy
from .environment import InventoryEnv, PeriodResult
from .executors import ExecutionRecord, FixedRouteExecutors, Route
from .features import StateFeatures
from .llm import StructuredLLM
from .or_policy import ORDecision, ORPolicy, ORPolicyConfig


@dataclass(frozen=True)
class InventoryBenchInstance:
    path: Path
    item_id: str
    trajectory_type: str
    lead_time_scenario: str
    demand_scenario: str
    initial_demand_history: tuple[float, ...]
    demands: tuple[float, ...]
    actual_lead_times: tuple[int | None, ...]
    expected_lead_time: int
    profit_per_unit: float
    holding_cost_per_unit: float
    critical_fractile: float
    contexts: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class FixedRunResult:
    instance: InventoryBenchInstance
    route: str
    decisions: tuple[ExecutionRecord, ...]
    periods: tuple[PeriodResult, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class RoutedRunResult:
    instance: InventoryBenchInstance
    method: str
    decisions: tuple[DynamicDecision, ...]
    periods: tuple[PeriodResult, ...]
    metrics: Mapping[str, float]


def _promised_lead_time(path: Path) -> tuple[str, int]:
    normalized = path.as_posix()
    if "lead_time_0" in normalized:
        return "lead_time_0", 0
    if "lead_time_4" in normalized:
        return "lead_time_4", 4
    if "lead_time_stochastic" in normalized:
        return "lead_time_stochastic", 2
    raise ValueError(f"cannot detect lead-time scenario from {path}")


def _actual_lead_time(value: Any) -> int | None:
    if isinstance(value, str) and value.strip().lower() == "inf":
        return None
    numeric = float(value)
    if np.isinf(numeric):
        return None
    if numeric < 0 or not numeric.is_integer():
        raise ValueError(f"invalid actual lead time: {value}")
    return int(numeric)


def load_inventorybench_instance(path: str | Path) -> InventoryBenchInstance:
    path = Path(path)
    train = pd.read_csv(path / "train.csv")
    test = pd.read_csv(path / "test.csv")
    demand_columns = [name for name in test.columns if name.startswith("demand_")]
    if len(demand_columns) != 1:
        raise ValueError(f"expected exactly one demand column in {path}")
    item_id = demand_columns[0][len("demand_") :]
    demand_col = f"demand_{item_id}"
    lead_col = f"lead_time_{item_id}"
    profit_col = f"profit_{item_id}"
    holding_col = f"holding_cost_{item_id}"
    required = {demand_col, lead_col, profit_col, holding_col}
    if not required.issubset(test.columns):
        raise ValueError(f"missing required columns: {sorted(required - set(test.columns))}")

    profit_values = test[profit_col].astype(float)
    holding_values = test[holding_col].astype(float)
    if profit_values.nunique() != 1 or holding_values.nunique() != 1:
        raise ValueError("M4 expects constant profit and holding cost within an instance")
    profit = float(profit_values.iloc[0])
    holding = float(holding_values.iloc[0])
    rho = profit / (profit + holding) if profit + holding > 0 else 0.5
    lead_scenario, promised_lead_time = _promised_lead_time(path)

    hidden_prefixes = ("demand_", "lead_time_")
    contexts: list[dict[str, Any]] = []
    for _, row in test.iterrows():
        context = {
            str(key): value.item() if hasattr(value, "item") else value
            for key, value in row.items()
            if not str(key).startswith(hidden_prefixes)
            and key not in {profit_col, holding_col}
            and not pd.isna(value)
        }
        contexts.append(context)

    parts = path.as_posix().split("/")
    trajectory_type = next(
        (part for part in parts if part in {"real_trajectory", "synthetic_trajectory"}),
        "unknown",
    )
    if trajectory_type == "synthetic_trajectory":
        demand_scenario = next((part for part in parts if part.startswith("p")), "unknown")
    else:
        demand_scenario = "real"

    return InventoryBenchInstance(
        path=path,
        item_id=item_id,
        trajectory_type=trajectory_type,
        lead_time_scenario=lead_scenario,
        demand_scenario=demand_scenario,
        initial_demand_history=tuple(float(value) for value in train[demand_col]),
        demands=tuple(float(value) for value in test[demand_col]),
        actual_lead_times=tuple(_actual_lead_time(value) for value in test[lead_col]),
        expected_lead_time=promised_lead_time,
        profit_per_unit=profit,
        holding_cost_per_unit=holding,
        critical_fractile=rho,
        contexts=tuple(contexts),
    )


def compute_metrics(periods: tuple[PeriodResult, ...], profit: float) -> dict[str, float]:
    total_demand = sum(row.demand for row in periods)
    total_sales = sum(row.sales for row in periods)
    total_lost = sum(row.lost_sales for row in periods)
    total_reward = sum(row.reward for row in periods)
    ideal_reward_upper_bound = profit * total_demand
    orders = [float(row.order_quantity) for row in periods]
    return {
        "normalized_reward": max(total_reward / ideal_reward_upper_bound, 0.0)
        if ideal_reward_upper_bound > 0 else 0.0,
        "total_reward": total_reward,
        "lost_sales_rate": total_lost / total_demand if total_demand else 0.0,
        "fill_rate": total_sales / total_demand if total_demand else 1.0,
        "average_inventory": mean(row.ending_inventory for row in periods),
        "stockout_period_rate": mean(row.lost_sales > 0 for row in periods),
        "order_quantity_std": pstdev(orders) if len(orders) > 1 else 0.0,
        "negative_profit": float(total_reward < 0),
        "llm_calls": 0.0,
    }


def _environment(instance: InventoryBenchInstance) -> InventoryEnv:
    return InventoryEnv(
        instance.demands,
        expected_lead_time=instance.expected_lead_time,
        actual_lead_times=instance.actual_lead_times,
        profit_per_unit=instance.profit_per_unit,
        holding_cost_per_unit=instance.holding_cost_per_unit,
        critical_fractile=instance.critical_fractile,
        initial_demand_history=instance.initial_demand_history,
        contexts=instance.contexts,
    )


def _restore_dynamic_decision(data: Mapping[str, Any]) -> DynamicDecision:
    """Rebuild one logged decision so an interrupted run can be replayed."""
    routing_data = dict(data["routing"])
    routing_data["selected_route"] = Route(routing_data["selected_route"])
    routing_data["reason_codes"] = tuple(routing_data.get("reason_codes", ()))
    routing_data["active_memory"] = tuple(
        MemoryItem(**item) for item in routing_data.get("active_memory", ())
    )
    execution_data = dict(data["execution"])
    execution_data["corrections"] = tuple(execution_data.get("corrections", ()))
    execution_data["or_decision"] = ORDecision(**execution_data["or_decision"])
    return DynamicDecision(
        features=StateFeatures(**data["features"]),
        routing=RoutingRecord(**routing_data),
        execution=ExecutionRecord(**execution_data),
        total_llm_calls=int(data["total_llm_calls"]),
    )


def _load_resume_rows(path: Path, horizon: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid resume log at {path}:{line_number}: {exc}"
            ) from exc
        expected_period = len(rows) + 1
        if row.get("period") != expected_period:
            raise ValueError(
                f"resume log period mismatch: expected {expected_period}, "
                f"found {row.get('period')}"
            )
        rows.append(row)
    if len(rows) > horizon:
        raise ValueError("resume log contains more periods than the instance horizon")
    return rows


def run_fixed_instance(
    instance: InventoryBenchInstance,
    route: Route | str,
    *,
    llm: StructuredLLM | None = None,
    decision_log: str | Path | None = None,
) -> FixedRunResult:
    env = _environment(instance)
    executors = FixedRouteExecutors(
        ORPolicy(ORPolicyConfig(cap_quantile=0.95)), llm
    )
    logger = DecisionLogger(decision_log) if decision_log else None
    decisions: list[ExecutionRecord] = []
    while not env.done:
        observation = env.observe()
        record = executors.execute(route, observation)
        decisions.append(record)
        if logger:
            logger.log(period=observation.period, record=record)
        env.step(record.order_quantity)
    metrics = compute_metrics(env.results, instance.profit_per_unit)
    metrics["llm_calls"] = float(sum(row.llm_calls for row in decisions))
    return FixedRunResult(
        instance=instance,
        route=route.value if isinstance(route, Route) else str(route),
        decisions=tuple(decisions),
        periods=env.results,
        metrics=metrics,
    )


def run_routed_instance(
    instance: InventoryBenchInstance,
    method: str,
    *,
    llm: StructuredLLM,
    controller_config: ControllerConfig | None = None,
    decision_log: str | Path | None = None,
    resume: bool = False,
    progress: Callable[[int, int, bool], None] | None = None,
) -> RoutedRunResult:
    method = method.upper()
    if method not in {"RULE_ROUTER", "SGIRA"}:
        raise ValueError("method must be RULE_ROUTER or SGIRA")
    env = _environment(instance)
    executors = FixedRouteExecutors(
        ORPolicy(ORPolicyConfig(cap_quantile=0.95)), llm
    )
    policy = (
        RuleRoutingPolicy(executors)
        if method == "RULE_ROUTER"
        else SGIRAPolicy(
            GeminiController(llm, config=controller_config), executors
        )
    )
    log_path = Path(decision_log) if decision_log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            log_path.write_text("", encoding="utf-8")
    decisions: list[DynamicDecision] = []

    resume_rows = _load_resume_rows(log_path, env.horizon) if log_path and resume else []
    for row in resume_rows:
        decision = _restore_dynamic_decision(row)
        observation = env.observe()
        if observation.period != row["period"]:
            raise ValueError("resume replay is not aligned with the environment")
        decisions.append(decision)
        env.step(decision.execution.order_quantity)
        if (
            method == "SGIRA"
            and decision.routing.source == "gemini"
            and policy.controller.config.include_memory
        ):
            memory_update = decision.routing.parsed_output.get("memory_update", "")
            policy.controller.memory.update(str(memory_update), observation.period)
        if progress:
            progress(observation.period, env.horizon, True)

    while not env.done:
        observation = env.observe()
        decision = policy.decide(observation)
        decisions.append(decision)
        if log_path:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"period": observation.period, **asdict(decision)},
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "\n")
        env.step(decision.execution.order_quantity)
        if progress:
            progress(observation.period, env.horizon, False)

    metrics = compute_metrics(env.results, instance.profit_per_unit)
    selected = [decision.routing.selected_route.value for decision in decisions]
    metrics["llm_calls"] = float(sum(
        decision.total_llm_calls for decision in decisions
    ))
    metrics["route_switches"] = float(sum(
        current != previous for previous, current in zip(selected, selected[1:])
    ))
    metrics["execution_fallbacks"] = float(sum(
        decision.execution.fallback for decision in decisions
    ))
    metrics["controller_fallbacks"] = float(sum(
        decision.routing.source in {"rule_fallback", "or_fallback"}
        for decision in decisions
    ))
    for route in Route:
        metrics[f"route_share_{route.value.lower()}"] = (
            selected.count(route.value) / len(selected) if selected else 0.0
        )
    return RoutedRunResult(
        instance=instance,
        method=method,
        decisions=tuple(decisions),
        periods=env.results,
        metrics=metrics,
    )


def save_fixed_run(result: FixedRunResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"period": row.period, "order_quantity": row.order_quantity}
        for row in result.periods
    ).to_csv(output_dir / "results.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(dict(result.metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def save_routed_run(result: RoutedRunResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "period": period.period,
            "order_quantity": decision.execution.order_quantity,
            "selected_route": decision.routing.selected_route.value,
            "routing_source": decision.routing.source,
        }
        for period, decision in zip(result.periods, result.decisions)
    ).to_csv(output_dir / "results.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(dict(result.metrics), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def summarize_runs(runs: list[FixedRunResult]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append({
            "route": run.route,
            "trajectory_type": run.instance.trajectory_type,
            "lead_time_scenario": run.instance.lead_time_scenario,
            "demand_scenario": run.instance.demand_scenario,
            **run.metrics,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    dimensions = ["route", "trajectory_type", "lead_time_scenario", "demand_scenario"]
    numeric = [name for name in frame.columns if name not in dimensions]
    return frame.groupby(dimensions, dropna=False)[numeric].mean().reset_index()


def summarize_routed_runs(runs: list[RoutedRunResult]) -> pd.DataFrame:
    rows = [{
        "method": run.method,
        "trajectory_type": run.instance.trajectory_type,
        "lead_time_scenario": run.instance.lead_time_scenario,
        "demand_scenario": run.instance.demand_scenario,
        **run.metrics,
    } for run in runs]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    dimensions = ["method", "trajectory_type", "lead_time_scenario", "demand_scenario"]
    numeric = [name for name in frame.columns if name not in dimensions]
    return frame.groupby(dimensions, dropna=False)[numeric].mean().reset_index()
