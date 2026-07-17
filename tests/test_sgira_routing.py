from __future__ import annotations

import unittest
from typing import Any, Mapping

from sgira.controller import ControllerConfig, GeminiController, ShortTermMemory
from sgira.dynamic_policy import RuleRoutingPolicy, SGIRAPolicy
from sgira.environment import InventoryEnv
from sgira.executors import FixedRouteExecutors, Route
from sgira.features import StateFeatures, build_state_features
from sgira.or_policy import ORPolicy
from sgira.rule_router import RuleRouter


class RoutingFakeLLM:
    def __init__(self, responses: Mapping[str, Any]) -> None:
        self.responses = dict(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(
        self, *, role: str, system_prompt: str, payload: Mapping[str, Any]
    ) -> Any:
        self.calls.append({"role": role, "system": system_prompt, "payload": payload})
        value = self.responses[role]
        if isinstance(value, Exception):
            raise value
        return value(payload) if callable(value) else value


def controller_output(**overrides: Any) -> dict[str, Any]:
    value = {
        "demand_regime": "stable",
        "supply_regime": "normal",
        "selected_route": "OR",
        "confidence": 0.9,
        "reason_codes": ["STABLE"],
        "short_reason": "stable",
        "memory_update": "",
    }
    value.update(overrides)
    return value


def feature_values(**overrides: Any) -> StateFeatures:
    value = dict(
        recent_demand_mean=10, recent_demand_std=1, cv=0.1, trend=0,
        shift=0, coverage=1, delay_score=0, oldest_order_age=0,
        overdue_order_count=0,
    )
    value.update(overrides)
    return StateFeatures(**value)


class FeatureAndRuleTests(unittest.TestCase):
    def test_features_detect_trend_shift_coverage_and_delay(self) -> None:
        env = InventoryEnv(
            [50, 0], expected_lead_time=1, actual_lead_times=[None, 1],
            initial_demand_history=[10, 10, 10, 20, 30, 40],
        )
        observation, _ = env.step(20)
        result = build_state_features(observation)
        self.assertGreater(result.trend, 0)
        self.assertGreater(result.shift, 0)
        self.assertGreater(result.coverage, 0)
        self.assertEqual(result.oldest_order_age, 1)
        self.assertAlmostEqual(result.delay_score, 1 / (1 + 1e-6))

    def test_rule_router_covers_all_four_routes(self) -> None:
        router = RuleRouter()
        self.assertEqual(router.select(feature_values()).route, Route.OR)
        self.assertEqual(
            router.select(feature_values(trend=0.1)).route, Route.LLM_TO_OR
        )
        self.assertEqual(
            router.select(feature_values(delay_score=2, overdue_order_count=1)).route,
            Route.OR_TO_LLM,
        )
        self.assertEqual(
            router.select(feature_values(
                trend=0.1, delay_score=2, overdue_order_count=1
            )).route,
            Route.LLM,
        )


class ControllerAndDynamicTests(unittest.TestCase):
    def observation(self):
        return InventoryEnv(
            [1], expected_lead_time=1, initial_demand_history=[10, 10, 10]
        ).observe()

    def test_high_confidence_route_and_bounded_memory(self) -> None:
        llm = RoutingFakeLLM({
            "controller": controller_output(
                selected_route="OR_TO_LLM", memory_update="supplier delayed"
            )
        })
        memory = ShortTermMemory(max_items=2, ttl_periods=2)
        record = GeminiController(llm, memory=memory).select(
            self.observation(), feature_values()
        )
        self.assertEqual(record.selected_route, Route.OR_TO_LLM)
        self.assertEqual(record.source, "gemini")
        memory.update("second", 2)
        memory.update("third", 2)
        self.assertEqual(len(memory.active(2)), 2)
        self.assertEqual(memory.active(5), ())

    def test_low_confidence_uses_rule_and_invalid_output_uses_or(self) -> None:
        low = RoutingFakeLLM({
            "controller": controller_output(confidence=0.2, selected_route="LLM")
        })
        low_record = GeminiController(low).select(
            self.observation(), feature_values(trend=0.1)
        )
        self.assertEqual(low_record.selected_route, Route.LLM_TO_OR)
        self.assertEqual(low_record.source, "rule_fallback")

        invalid = RoutingFakeLLM({"controller": "not-json"})
        invalid_record = GeminiController(invalid).select(
            self.observation(), feature_values(trend=0.1)
        )
        self.assertEqual(invalid_record.selected_route, Route.OR)
        self.assertEqual(invalid_record.source, "or_fallback")

    def test_controller_api_error_falls_back_to_or(self) -> None:
        llm = RoutingFakeLLM({"controller": RuntimeError("temporary outage")})
        record = GeminiController(llm).select(self.observation(), feature_values())
        self.assertEqual(record.selected_route, Route.OR)
        self.assertIn("temporary outage", record.fallback_reason)

    def test_ablation_payloads_remove_only_selected_information(self) -> None:
        llm = RoutingFakeLLM({"controller": controller_output()})
        controller = GeminiController(
            llm,
            config=ControllerConfig(
                include_context=False,
                include_memory=False,
                include_supply_diagnosis=False,
            ),
        )
        controller.select(self.observation(), feature_values(delay_score=2))
        payload = llm.calls[0]["payload"]
        self.assertEqual(payload["observable_state"]["context"], {})
        self.assertEqual(payload["observable_state"]["outstanding_orders"], [])
        self.assertEqual(payload["memory"], [])
        self.assertNotIn("delay_score", payload["features"])
        self.assertIn("trend", payload["features"])

    def test_sgira_separates_controller_and_executor_calls(self) -> None:
        llm = RoutingFakeLLM({
            "controller": controller_output(selected_route="LLM"),
            "llm_direct": {
                "order_quantity": 7, "reason_codes": [], "short_reason": "ok"
            },
        })
        executors = FixedRouteExecutors(ORPolicy(), llm)
        policy = SGIRAPolicy(GeminiController(llm), executors)
        decision = policy.decide(self.observation())
        self.assertIs(policy.executors, executors)
        self.assertEqual(decision.execution.order_quantity, 7)
        self.assertEqual(decision.total_llm_calls, 2)
        self.assertEqual(
            [call["role"] for call in llm.calls], ["controller", "llm_direct"]
        )

    def test_rule_policy_reuses_fixed_executor(self) -> None:
        llm = RoutingFakeLLM({
            "llm_to_or": {
                "effective_lead_time": 1, "protection_horizon": 2,
                "protection_demand_mean": 20, "protection_demand_std": 2,
                "reason_codes": [],
            }
        })
        env = InventoryEnv(
            [1], expected_lead_time=1, initial_demand_history=[1, 2, 3, 4, 5]
        )
        executors = FixedRouteExecutors(ORPolicy(), llm)
        policy = RuleRoutingPolicy(executors)
        decision = policy.decide(env.observe())
        self.assertIs(policy.executors, executors)
        self.assertEqual(decision.routing.selected_route, Route.LLM_TO_OR)
        self.assertEqual(decision.total_llm_calls, 1)


if __name__ == "__main__":
    unittest.main()
