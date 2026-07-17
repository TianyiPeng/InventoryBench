from __future__ import annotations

import unittest

from sgira.environment import InventoryEnv


class InventoryEnvironmentTests(unittest.TestCase):
    def test_five_period_hand_calculation(self) -> None:
        env = InventoryEnv(
            [4, 6, 3, 8, 2],
            initial_inventory=5,
            expected_lead_time=1,
            actual_lead_times=[1, 1, 1, 1, 1],
            profit_per_unit=10,
            holding_cost_per_unit=1,
        )
        expected = [
            (3, 0, 4, 0, 1, 39),
            (5, 3, 4, 2, 0, 40),
            (2, 5, 3, 0, 2, 28),
            (4, 2, 4, 4, 0, 40),
            (0, 4, 2, 0, 2, 18),
        ]
        for order, arrivals, sales, lost, ending, reward in expected:
            _, result = env.step(order)
            self.assertEqual(result.arrivals, arrivals)
            self.assertEqual(result.sales, sales)
            self.assertEqual(result.lost_sales, lost)
            self.assertEqual(result.ending_inventory, ending)
            self.assertEqual(result.reward, reward)
        self.assertTrue(env.done)
        self.assertEqual(sum(row.reward for row in env.results), 165)

    def test_zero_lead_time_order_arrives_before_current_demand(self) -> None:
        env = InventoryEnv([5], actual_lead_times=[0], profit_per_unit=2)
        _, result = env.step(5)
        self.assertEqual(result.arrivals, 5)
        self.assertEqual(result.sales, 5)
        self.assertEqual(result.lost_sales, 0)

    def test_lost_order_remains_observably_in_transit(self) -> None:
        env = InventoryEnv(
            [0, 0, 0], expected_lead_time=1, actual_lead_times=[None, 1, 1]
        )
        observation, _ = env.step(7)
        self.assertEqual(observation.in_transit_total, 7)
        self.assertEqual(observation.outstanding_orders[0].age, 1)
        observation, _ = env.step(0)
        self.assertEqual(observation.in_transit_total, 7)
        self.assertEqual(observation.outstanding_orders[0].age, 2)

    def test_observation_does_not_reveal_current_or_future_outcomes(self) -> None:
        env = InventoryEnv(
            [99, 123],
            expected_lead_time=2,
            actual_lead_times=[None, 1],
            contexts=[{"promotion": True}, {"promotion": False}],
        )
        observation = env.reset()
        public_fields = set(observation.__dataclass_fields__)
        self.assertNotIn("demand", public_fields)
        self.assertNotIn("future_demand", public_fields)
        self.assertNotIn("actual_lead_time", public_fields)
        self.assertNotIn("arrival_period", public_fields)
        self.assertNotIn("is_lost", public_fields)
        self.assertEqual(observation.demand_history, ())
        self.assertEqual(observation.context, {"promotion": True})

    def test_invalid_orders_are_rejected(self) -> None:
        env = InventoryEnv([1], order_cap=3)
        with self.assertRaises(TypeError):
            env.step(1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            env.step(-1)
        with self.assertRaises(ValueError):
            env.step(4)


if __name__ == "__main__":
    unittest.main()
