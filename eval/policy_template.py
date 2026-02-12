"""
Policy Template for InventoryBench Submissions

This template defines the interface that all policies must implement.
Your policy receives:
  1. Initial context: Historical samples + promised lead time
  2. Per-period state: Current inventory, in-transit orders, period info
  3. Per-period metadata: Profit, holding cost, dates (but NOT actual lead times)

Output: Order quantity for current period
"""

from typing import Dict, List, Tuple


class InventoryPolicy:
    """
    Base class for inventory policies.
    
    Participants should subclass this and implement the two required methods.
    """
    
    def __init__(
        self,
        item_id: str,
        initial_samples: List[Tuple[str, float]],  # [(date, demand), ...]
        promised_lead_time: int,
        profit_per_unit: float,
        holding_cost_per_unit: float,
        product_description: str = None
    ):
        """
        Initialize policy with instance context.
        
        Args:
            item_id: SKU identifier
            initial_samples: Historical demand samples from train.csv
                            Format: [(date_str, demand), ...]
            promised_lead_time: Supplier-promised lead time (0, 2, or 4 periods)
            profit_per_unit: Profit earned per unit sold
            holding_cost_per_unit: Cost per unit held in inventory per period
            product_description: Optional product description (available for real trajectories)
        """
        self.item_id = item_id
        self.initial_samples = initial_samples
        self.promised_lead_time = promised_lead_time
        self.profit_per_unit = profit_per_unit
        self.holding_cost_per_unit = holding_cost_per_unit
        self.product_description = product_description
        
        # Extract historical demand values
        self.historical_demands = [demand for _, demand in initial_samples]
        
        # Initialize any policy-specific state
        self.reset()
    
    def reset(self):
        """
        Reset policy state for new episode.
        
        Called before starting a new instance.
        Override this to reset any internal state.
        """
        pass
    
    def get_order(
        self,
        period: int,
        current_date: str,
        on_hand_inventory: float,
        in_transit_total: float,
        previous_demand: float,  # Actual demand from previous period (0 if period==1)
        previous_order: float,   # Your order from previous period (0 if period==1)
        previous_arrivals: float,  # Units that arrived last period (0 if period==1)
        profit_per_unit: float,
        holding_cost_per_unit: float
    ) -> float:
        """
        Determine order quantity for current period.
        
        Args:
            period: Current period number (1-indexed)
            current_date: Date string for current period (e.g., "2019-07-01")
            on_hand_inventory: Current on-hand inventory level
            in_transit_total: Total units in transit (not yet arrived)
            previous_demand: Actual demand observed in previous period
            previous_order: Order quantity you placed in previous period
            previous_arrivals: Units that arrived in previous period
            profit_per_unit: Current profit per unit (may change per period)
            holding_cost_per_unit: Current holding cost (may change per period)
        
        Returns:
            order_quantity: Non-negative integer order quantity
            
        Note: You do NOT observe actual lead times - must infer from arrivals!
        """
        raise NotImplementedError("Subclass must implement get_order()")


# ============================================================================
# Example Policy: Constant Order Quantity
# ============================================================================

class ExamplePolicy(InventoryPolicy):
    """
    Minimal example policy that always orders quantity 1.

    This is provided as a template for participants to understand the interface.
    Real policies should implement more sophisticated logic!
    """

    def get_order(
        self,
        period: int,
        current_date: str,
        on_hand_inventory: float,
        in_transit_total: float,
        previous_demand: float,
        previous_order: float,
        previous_arrivals: float,
        profit_per_unit: float,
        holding_cost_per_unit: float
    ) -> float:
        """Always order quantity 1 (example only - not a good policy!)"""
        return 1
