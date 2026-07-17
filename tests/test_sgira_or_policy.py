from __future__ import annotations

import math
import unittest

from scipy.stats import norm

from sgira.environment import InventoryEnv
from sgira.or_policy import ORPolicy, ORPolicyConfig


class ORPolicyTests(unittest.TestCase):
    def test_deterministic_demand_base_stock(self) -> None:
        env = InventoryEnv(
            [1], initial_inventory=3, expected_lead_time=2,
            critical_fractile=0.8, initial_demand_history=[10, 10, 10],
        )
        decision = ORPolicy().decide(env.observe())
        self.assertEqual(decision.protection_horizon, 3)
        self.assertEqual(decision.target_inventory, 30)
        self.assertEqual(decision.inventory_position, 3)
        self.assertEqual(decision.order_quantity, 27)

    def test_sample_standard_deviation_and_rounding(self) -> None:
        env = InventoryEnv(
            [1], expected_lead_time=1, critical_fractile=0.8,
            initial_demand_history=[8, 10, 12],
        )
        decision = ORPolicy().decide(env.observe())
        expected_target = 20 + float(norm.ppf(0.8)) * 2 * math.sqrt(2)
        self.assertAlmostEqual(decision.demand_std, 2.0)
        self.assertAlmostEqual(decision.target_inventory, expected_target)
        self.assertEqual(decision.order_quantity, math.ceil(expected_target))

    def test_inventory_position_includes_outstanding_orders(self) -> None:
        env = InventoryEnv(
            [10, 0], expected_lead_time=1, actual_lead_times=[None, 1],
            initial_demand_history=[10, 10],
        )
        observation, _ = env.step(6)
        decision = ORPolicy().decide(observation)
        self.assertEqual(decision.inventory_position, 6)
        self.assertEqual(decision.order_quantity, 14)

    def test_order_cap_and_history_window(self) -> None:
        env = InventoryEnv(
            [1], expected_lead_time=0, critical_fractile=0.5,
            initial_demand_history=[100, 10, 10],
        )
        decision = ORPolicy(
            ORPolicyConfig(order_cap=7, history_window=2)
        ).decide(env.observe())
        self.assertEqual(decision.demand_mean, 10)
        self.assertEqual(decision.uncapped_order_quantity, 10)
        self.assertEqual(decision.order_quantity, 7)

    def test_no_history_is_safe(self) -> None:
        decision = ORPolicy().decide(InventoryEnv([1]).observe())
        self.assertEqual(decision.order_quantity, 0)

    def test_inventorybench_dynamic_cap(self) -> None:
        env = InventoryEnv(
            [1], expected_lead_time=4, critical_fractile=0.8,
            initial_demand_history=[8, 10, 12],
        )
        decision = ORPolicy(ORPolicyConfig(cap_quantile=0.95)).decide(env.observe())
        expected_cap = math.ceil(10 + float(norm.ppf(0.95)) * 2)
        self.assertEqual(decision.computed_order_cap, expected_cap)
        self.assertEqual(decision.order_quantity, expected_cap)


if __name__ == "__main__":
    unittest.main()
