from __future__ import annotations

import unittest

from sgira.schemas import (
    ControllerOutput, LLMDirectOutput, LLMToOROutput, ORToLLMOutput, SchemaError,
)


class SchemaTests(unittest.TestCase):
    def test_valid_outputs_parse(self) -> None:
        self.assertEqual(
            LLMDirectOutput.parse(
                '{"order_quantity": 7, "reason_codes": [], "short_reason": "ok"}'
            ).order_quantity,
            7,
        )
        self.assertEqual(
            ORToLLMOutput.parse({
                "decision": "accept", "or_order_quantity": 7,
                "final_order_quantity": 7, "reason_codes": [],
            }).decision,
            "accept",
        )
        self.assertEqual(
            LLMToOROutput.parse({
                "effective_lead_time": 1, "protection_horizon": 2,
                "protection_demand_mean": 10, "protection_demand_std": 2,
                "reason_codes": [],
            }).protection_horizon,
            2,
        )

    def test_invalid_and_unsafe_values_are_rejected(self) -> None:
        bad_values = [
            "not-json",
            {"order_quantity": -1, "reason_codes": [], "short_reason": "bad"},
            {"order_quantity": 1.5, "reason_codes": [], "short_reason": "bad"},
        ]
        for value in bad_values:
            with self.assertRaises(SchemaError):
                LLMDirectOutput.parse(value)
        with self.assertRaises(SchemaError):
            LLMToOROutput.parse({
                "effective_lead_time": 1, "protection_horizon": 0,
                "protection_demand_mean": 10, "protection_demand_std": 2,
                "reason_codes": [],
            })

    def test_controller_schema_is_strict(self) -> None:
        valid = {
            "demand_regime": "stable", "supply_regime": "normal",
            "selected_route": "OR", "confidence": 0.8,
            "reason_codes": [], "short_reason": "ok", "memory_update": "",
        }
        self.assertEqual(ControllerOutput.parse(valid).selected_route, "OR")
        for field, value in (("selected_route", "BAD"), ("confidence", 1.1)):
            with self.assertRaises(SchemaError):
                ControllerOutput.parse({**valid, field: value})

if __name__ == "__main__":
    unittest.main()
