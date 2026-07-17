from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from sgira.environment import InventoryEnv
from sgira.executors import FixedRouteExecutors, Route, observation_payload
from sgira.decision_logger import DecisionLogger
from sgira.or_policy import ORPolicy, ORPolicyConfig


class FakeLLM:
    def __init__(self, responses: Mapping[str, Any] | None = None) -> None:
        self.responses = dict(responses or {})
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self, *, role: str, system_prompt: str, payload: Mapping[str, Any]
    ) -> Any:
        self.calls.append({"role": role, "system": system_prompt, "payload": payload})
        response = self.responses.get(role)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(payload)
        return response


def base_env(*, order_cap: int | None = None) -> InventoryEnv:
    return InventoryEnv(
        [4, 5, 6],
        initial_inventory=0,
        expected_lead_time=0,
        actual_lead_times=[0, 0, 0],
        initial_demand_history=[10, 10, 10],
        critical_fractile=0.5,
        order_cap=order_cap,
    )


class ExecutorTests(unittest.TestCase):
    def test_direct_llm_is_capped_and_logged(self) -> None:
        llm = FakeLLM({
            "llm_direct": {
                "order_quantity": 99,
                "reason_codes": ["UPWARD_TREND"],
                "short_reason": "trend",
            }
        })
        executor = FixedRouteExecutors(
            ORPolicy(ORPolicyConfig(order_cap=12)), llm
        )
        record = executor.execute(Route.LLM, base_env(order_cap=12).observe())
        self.assertEqual(record.order_quantity, 12)
        self.assertEqual(record.corrections, ("ORDER_CAP_APPLIED",))
        self.assertFalse(record.fallback)
        self.assertEqual(record.llm_calls, 1)

    def test_or_to_llm_adjustment_is_bounded(self) -> None:
        llm = FakeLLM({
            "or_to_llm": {
                "decision": "increase",
                "or_order_quantity": 10,
                "final_order_quantity": 100,
                "reason_codes": ["PROMOTION_ACTIVE"],
            }
        })
        record = FixedRouteExecutors(ORPolicy(), llm).execute(
            Route.OR_TO_LLM, base_env().observe()
        )
        self.assertEqual(record.order_quantity, 13)
        self.assertIn("OR_ADJUSTMENT_BOUND_APPLIED", record.corrections)

    def test_llm_to_or_uses_llm_moments_in_deterministic_formula(self) -> None:
        llm = FakeLLM({
            "llm_to_or": {
                "effective_lead_time": 2,
                "protection_horizon": 3,
                "protection_demand_mean": 25,
                "protection_demand_std": 7,
                "reason_codes": ["UPWARD_TREND"],
            }
        })
        env = InventoryEnv(
            [1], initial_inventory=5, critical_fractile=0.5,
            initial_demand_history=[10, 10],
        )
        record = FixedRouteExecutors(ORPolicy(), llm).execute(
            Route.LLM_TO_OR, env.observe()
        )
        self.assertEqual(record.order_quantity, 20)
        self.assertEqual(record.parsed_output["target_inventory"], 25)

    def test_invalid_json_falls_back_to_or_and_preserves_raw_output(self) -> None:
        llm = FakeLLM({"llm_direct": "not-json"})
        record = FixedRouteExecutors(ORPolicy(), llm).execute(
            Route.LLM, base_env().observe()
        )
        self.assertTrue(record.fallback)
        self.assertEqual(record.applied_route, Route.OR.value)
        self.assertEqual(record.order_quantity, 10)
        self.assertEqual(record.raw_output, "not-json")
        self.assertEqual(record.llm_request["role"], "llm_direct")
        self.assertIn("system_prompt", record.llm_request)

    def test_api_error_and_invalid_route_fall_back_without_interrupting(self) -> None:
        llm = FakeLLM({"llm_direct": RuntimeError("temporary outage")})
        executor = FixedRouteExecutors(ORPolicy(), llm)
        api_record = executor.execute(Route.LLM, base_env().observe())
        route_record = executor.execute("UNKNOWN", base_env().observe())
        self.assertTrue(api_record.fallback)
        self.assertIn("temporary outage", api_record.fallback_reason)
        self.assertEqual(api_record.llm_calls, 1)
        self.assertTrue(route_record.fallback)
        self.assertEqual(route_record.llm_calls, 0)

    def test_decision_logger_saves_prompt_input_raw_and_parsed_output(self) -> None:
        llm = FakeLLM({
            "llm_direct": {
                "order_quantity": 7,
                "reason_codes": ["STABLE"],
                "short_reason": "stable",
            }
        })
        record = FixedRouteExecutors(ORPolicy(), llm).execute(
            Route.LLM, base_env().observe()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision_log.jsonl"
            DecisionLogger(path).log(period=1, record=record)
            row = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(row["period"], 1)
        self.assertEqual(row["llm_request"]["role"], "llm_direct")
        self.assertEqual(row["raw_output"]["order_quantity"], 7)
        self.assertEqual(row["parsed_output"]["order_quantity"], 7)

    def test_echo_mismatch_falls_back_to_or(self) -> None:
        llm = FakeLLM({
            "or_to_llm": {
                "decision": "accept",
                "or_order_quantity": 999,
                "final_order_quantity": 999,
                "reason_codes": [],
            }
        })
        record = FixedRouteExecutors(ORPolicy(), llm).execute(
            Route.OR_TO_LLM, base_env().observe()
        )
        self.assertTrue(record.fallback)
        self.assertEqual(record.order_quantity, 10)

    def test_all_fixed_routes_finish_an_episode(self) -> None:
        responses = {
            "llm_direct": {
                "order_quantity": 8,
                "reason_codes": ["STABLE"],
                "short_reason": "stable",
            },
            "or_to_llm": lambda payload: {
                "decision": "accept",
                "or_order_quantity": payload["or_decision"]["order_quantity"],
                "final_order_quantity": payload["or_decision"]["order_quantity"],
                "reason_codes": ["OR_VALID"],
            },
            "llm_to_or": {
                "effective_lead_time": 0,
                "protection_horizon": 1,
                "protection_demand_mean": 8,
                "protection_demand_std": 1,
                "reason_codes": ["STABLE"],
            },
        }
        for route in Route:
            env = base_env(order_cap=20)
            executor = FixedRouteExecutors(
                ORPolicy(ORPolicyConfig(order_cap=20)), FakeLLM(responses)
            )
            while not env.done:
                record = executor.execute(route, env.observe())
                self.assertGreaterEqual(record.order_quantity, 0)
                env.step(record.order_quantity)
            self.assertEqual(len(env.results), 3)

    def test_payload_contains_no_hidden_outcomes(self) -> None:
        payload = observation_payload(base_env().observe())
        self.assertNotIn("current_demand", payload)
        self.assertNotIn("future_demand", payload)
        self.assertNotIn("actual_lead_times", payload)
        self.assertNotIn("arrival_period", str(payload))
        self.assertNotIn("is_lost", str(payload))


if __name__ == "__main__":
    unittest.main()
