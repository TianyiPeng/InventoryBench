from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from sgira.executors import Route
from sgira.inventorybench import (
    load_inventorybench_instance,
    run_routed_instance,
    save_routed_run,
    summarize_routed_runs,
)


ROOT = Path(__file__).resolve().parent.parent
INSTANCE = (
    ROOT / "benchmark" / "synthetic_trajectory" / "lead_time_stochastic"
    / "p04_increasing_trend" / "v1_linear_100t" / "r1_high"
)


class CyclingFakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self, *, role: str, system_prompt: str, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls.append({"role": role, "system": system_prompt, "payload": payload})
        if role == "controller":
            period = payload["observable_state"]["period"]
            route = (Route.OR, Route.LLM, Route.OR_TO_LLM, Route.LLM_TO_OR)[
                (period - 1) % 4
            ]
            return {
                "demand_regime": "trend_up",
                "supply_regime": "normal",
                "selected_route": route.value,
                "confidence": 0.9,
                "reason_codes": ["UPWARD_TREND"],
                "short_reason": "test route",
                "memory_update": "demand remains elevated" if period == 1 else "",
            }
        if role == "llm_direct":
            return {
                "order_quantity": 20,
                "reason_codes": ["UPWARD_TREND"],
                "short_reason": "test order",
            }
        if role == "or_to_llm":
            quantity = payload["or_decision"]["order_quantity"]
            return {
                "decision": "accept",
                "or_order_quantity": quantity,
                "final_order_quantity": quantity,
                "reason_codes": ["OR_VALID"],
            }
        history = payload["demand_history"][-5:]
        demand_mean = sum(history) / len(history)
        horizon = payload["expected_lead_time"] + 1
        return {
            "effective_lead_time": payload["expected_lead_time"],
            "protection_horizon": horizon,
            "protection_demand_mean": demand_mean * horizon,
            "protection_demand_std": 10,
            "reason_codes": ["UPWARD_TREND"],
        }


class DynamicInventoryBenchTests(unittest.TestCase):
    def test_sgira_completes_instance_and_records_route_behavior(self) -> None:
        instance = load_inventorybench_instance(INSTANCE)
        llm = CyclingFakeLLM()
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "decision_log.jsonl"
            result = run_routed_instance(
                instance, "SGIRA", llm=llm, decision_log=log_path
            )
            rows = [
                json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
            ]
            save_routed_run(result, Path(directory) / "saved")
            self.assertTrue((Path(directory) / "saved" / "results.csv").exists())
        self.assertEqual(len(result.periods), len(instance.demands))
        self.assertEqual(len(rows), len(instance.demands))
        self.assertGreater(result.metrics["route_switches"], 0)
        shares = sum(
            result.metrics[f"route_share_{route.value.lower()}"] for route in Route
        )
        self.assertAlmostEqual(shares, 1.0)
        self.assertGreater(result.metrics["llm_calls"], len(instance.demands))

    def test_rule_router_completes_with_the_same_executors(self) -> None:
        instance = load_inventorybench_instance(INSTANCE)
        result = run_routed_instance(instance, "RULE_ROUTER", llm=CyclingFakeLLM())
        self.assertEqual(len(result.periods), len(instance.demands))
        self.assertTrue(all(row.order_quantity >= 0 for row in result.periods))
        self.assertEqual(result.metrics["controller_fallbacks"], 0)
        summary = summarize_routed_runs([result])
        self.assertEqual(summary.iloc[0]["method"], "RULE_ROUTER")

    def test_controller_never_receives_hidden_trajectory_values(self) -> None:
        instance = load_inventorybench_instance(INSTANCE)
        llm = CyclingFakeLLM()
        run_routed_instance(instance, "SGIRA", llm=llm)
        controller_calls = [call for call in llm.calls if call["role"] == "controller"]
        self.assertEqual(len(controller_calls), len(instance.demands))
        for call in controller_calls:
            serialized = json.dumps(call["payload"], ensure_ascii=False)
            self.assertNotIn("actual_lead_time", serialized)
            self.assertNotIn("future_demand", serialized)
            self.assertNotIn("arrival_period", serialized)
            self.assertNotIn("is_lost", serialized)

    def test_resume_replays_saved_periods_without_repeating_model_calls(self) -> None:
        instance = load_inventorybench_instance(INSTANCE)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete_log = root / "complete.jsonl"
            baseline = run_routed_instance(
                instance, "SGIRA", llm=CyclingFakeLLM(), decision_log=complete_log
            )
            complete_lines = complete_log.read_text(encoding="utf-8").splitlines()
            interrupted_log = root / "interrupted.jsonl"
            interrupted_log.write_text(
                "\n".join(complete_lines[:7]) + "\n", encoding="utf-8"
            )

            resumed_llm = CyclingFakeLLM()
            resumed = run_routed_instance(
                instance,
                "SGIRA",
                llm=resumed_llm,
                decision_log=interrupted_log,
                resume=True,
            )

            self.assertEqual(resumed.periods, baseline.periods)
            self.assertEqual(resumed.metrics, baseline.metrics)
            self.assertEqual(
                [row.execution.order_quantity for row in resumed.decisions],
                [row.execution.order_quantity for row in baseline.decisions],
            )
            self.assertEqual(
                len(interrupted_log.read_text(encoding="utf-8").splitlines()),
                len(instance.demands),
            )
            controller_calls = [
                call for call in resumed_llm.calls if call["role"] == "controller"
            ]
            self.assertEqual(
                controller_calls[0]["payload"]["observable_state"]["period"], 8
            )


if __name__ == "__main__":
    unittest.main()
