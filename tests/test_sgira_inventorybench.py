from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from sgira.executors import Route
from sgira.inventorybench import (
    compute_metrics,
    load_inventorybench_instance,
    run_fixed_instance,
    summarize_runs,
)


ROOT = Path(__file__).resolve().parent.parent
RELATIVE = Path("p01_stationary_iid/v1_normal_100_25/r1_high")


class FixedFakeLLM:
    def complete_json(
        self, *, role: str, system_prompt: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        del system_prompt
        if role == "llm_direct":
            return {"order_quantity": 10, "reason_codes": [], "short_reason": "ok"}
        if role == "or_to_llm":
            quantity = payload["or_decision"]["order_quantity"]
            return {
                "decision": "accept", "or_order_quantity": quantity,
                "final_order_quantity": quantity, "reason_codes": [],
            }
        return {
            "effective_lead_time": payload["expected_lead_time"],
            "protection_horizon": payload["expected_lead_time"] + 1,
            "protection_demand_mean": 20,
            "protection_demand_std": 2,
            "reason_codes": [],
        }


class InventoryBenchIntegrationTests(unittest.TestCase):
    def instance_path(self, lead: str) -> Path:
        return ROOT / "benchmark" / "synthetic_trajectory" / lead / RELATIVE

    def test_loader_hides_demand_and_actual_lead_time_from_context(self) -> None:
        instance = load_inventorybench_instance(self.instance_path("lead_time_stochastic"))
        self.assertEqual(instance.expected_lead_time, 2)
        self.assertEqual(len(instance.initial_demand_history), 5)
        self.assertEqual(len(instance.demands), len(instance.contexts))
        for context in instance.contexts:
            self.assertFalse(any(key.startswith("demand_") for key in context))
            self.assertFalse(any(key.startswith("lead_time_") for key in context))

    def test_or_matches_published_inventorybench_orders_period_by_period(self) -> None:
        official_root = ROOT / "results" / "OR (capped base stock)" / "results" / "synthetic_trajectory"
        for lead in ("lead_time_0", "lead_time_4", "lead_time_stochastic"):
            instance = load_inventorybench_instance(self.instance_path(lead))
            result = run_fixed_instance(instance, Route.OR)
            actual = [row.order_quantity for row in result.periods]
            expected = pd.read_csv(
                official_root / lead / RELATIVE / "results.csv"
            )["order_quantity"].astype(int).tolist()
            self.assertEqual(actual, expected, lead)

    def test_four_fixed_routes_run_to_completion_and_summarize(self) -> None:
        instance = load_inventorybench_instance(self.instance_path("lead_time_0"))
        runs = [
            run_fixed_instance(instance, route, llm=FixedFakeLLM())
            for route in Route
        ]
        self.assertTrue(all(len(run.periods) == len(instance.demands) for run in runs))
        self.assertEqual(set(summarize_runs(runs)["route"]), {route.value for route in Route})

    def test_metrics_use_ideal_reward_upper_bound(self) -> None:
        instance = load_inventorybench_instance(self.instance_path("lead_time_0"))
        result = run_fixed_instance(instance, Route.OR)
        metrics = compute_metrics(result.periods, instance.profit_per_unit)
        expected = max(
            sum(row.reward for row in result.periods)
            / (instance.profit_per_unit * sum(row.demand for row in result.periods)),
            0,
        )
        self.assertAlmostEqual(metrics["normalized_reward"], expected)
        self.assertAlmostEqual(
            metrics["fill_rate"] + metrics["lost_sales_rate"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
