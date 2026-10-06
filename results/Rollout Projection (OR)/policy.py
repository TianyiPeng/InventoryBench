"""Frozen JointRolloutProjection policy; generated-only configuration selection.
Independent implementation informed by SabreAgent, not its authors artifact.
ORAgent adapted from TianyiPeng/InventoryBench (MIT license, see repository LICENSE).
No files, paths, future demand or hidden supply are read at runtime.
"""

import io
from contextlib import redirect_stdout
import numpy as np
from scipy.stats import norm
_safe_print=lambda *args, **kwargs: None

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


class ORAgent:
    """
    OR algorithm baseline agent using base-stock policy.
    
    Supports two policies:
    1. Vanilla: order_quantity = max(base_stock - current_inventory, 0)
       where base_stock = μ̂ + z*σ̂
    2. Capped: order_quantity = max(min(base_stock - current_inventory, cap), 0)
       where cap = μ̂/(1+L) + Φ^(-1)(0.95) × σ̂/√(1+L)
    
    μ̂ = (1+L) × empirical_mean
    σ̂ = sqrt(1+L) × empirical_std
    z* = Φ^(-1)(q), where q = profit/(profit + holding_cost)
    """

    def __init__(self, items_config: dict, initial_samples: dict=None, policy: str='capped'):
        """
        Args:
            items_config: Dict of {item_id: {'lead_time': L, 'profit': p, 'holding_cost': h}}
            initial_samples: Optional dict of {item_id: [list of initial demand samples]}
                           If None or empty, will use only observed demands
            policy: 'vanilla' or 'capped' (default: 'capped')
        """
        self.items_config = items_config
        self.initial_samples = initial_samples if initial_samples else {}
        self.policy = policy
        self.observed_demands = {item_id: [] for item_id in items_config}
        print(f'\n=== OR Agent Initialized (Policy: {policy.upper()}) ===')
        for item_id, config in items_config.items():
            L = config['lead_time']
            p = config['profit']
            h = config['holding_cost']
            q = p / (p + h)
            z_star = norm.ppf(q)
            samples = self.initial_samples.get(item_id, [])
            print(f'{item_id}:')
            print(f'  Lead time (L): {L}')
            print(f'  Profit (p): {p}, Holding cost (h): {h}')
            print(f'  Critical fractile (q): {q:.4f}')
            _safe_print(f'  z* = Φ^(-1)(q): {z_star:.4f}')
            print(f"  Initial samples: {(samples if samples else 'None (will learn from observed demands)')}")

    def _parse_inventory_from_observation(self, observation: str, item_id: str) -> int:
        """
        Parse current total inventory (on-hand + in-transit) from observation.
        
        Observation format:
          chips(Regular) (...): Profit=$2/unit, Holding=$1/unit/period
            On-hand: 5, In-transit: 10 units
        
        Returns:
            Total inventory across all pipeline stages (on-hand + in-transit)
        """
        try:
            lines = observation.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith(f'{item_id}'):
                    if i + 1 < len(lines):
                        inventory_line = lines[i + 1]
                        if 'On-hand:' in inventory_line and 'In-transit:' in inventory_line:
                            on_hand_start = inventory_line.find('On-hand:') + len('On-hand:')
                            on_hand_end = inventory_line.find(',', on_hand_start)
                            on_hand = int(inventory_line[on_hand_start:on_hand_end].strip())
                            in_transit_start = inventory_line.find('In-transit:') + len('In-transit:')
                            in_transit_str = inventory_line[in_transit_start:].strip()
                            in_transit_str = in_transit_str.replace('units', '').strip()
                            in_transit = int(in_transit_str)
                            total_inventory = on_hand + in_transit
                            return total_inventory
            print(f'Warning: Could not parse inventory for {item_id}, assuming 0')
            return 0
        except Exception as e:
            print(f'Error parsing inventory for {item_id}: {e}')
            return 0

    def _calculate_order(self, item_id: str, current_inventory: int) -> dict:
        """
        Calculate order quantity using OR base-stock policy.
        
        Args:
            item_id: Item identifier
            current_inventory: Current total inventory (on-hand + in-transit)
            
        Returns:
            Dict with keys: order, empirical_mean, empirical_std, mu_hat, sigma_hat, 
                           L, z_star, q, base_stock, cap (if capped), order_uncapped (if capped)
        """
        config = self.items_config[item_id]
        L = config['lead_time']
        p = config['profit']
        h = config['holding_cost']
        initial = self.initial_samples.get(item_id, [])
        all_samples = initial + self.observed_demands[item_id]
        if not all_samples:
            return {'order': 0, 'empirical_mean': 0, 'empirical_std': 0, 'mu_hat': 0, 'sigma_hat': 0, 'L': L, 'z_star': 0, 'q': p / (p + h), 'base_stock': 0, 'current_inventory': current_inventory}
        empirical_mean = np.mean(all_samples)
        empirical_std = np.std(all_samples, ddof=1) if len(all_samples) > 1 else 0
        L_for_calc = L if not np.isinf(L) else 1000
        mu_hat = (1 + L_for_calc) * empirical_mean
        sigma_hat = np.sqrt(1 + L_for_calc) * empirical_std
        q = p / (p + h)
        z_star = norm.ppf(q)
        base_stock = mu_hat + z_star * sigma_hat
        result = {'empirical_mean': empirical_mean, 'empirical_std': empirical_std, 'mu_hat': mu_hat, 'sigma_hat': sigma_hat, 'L': L, 'z_star': z_star, 'q': q, 'base_stock': base_stock, 'current_inventory': current_inventory}
        if self.policy == 'vanilla':
            order = max(int(np.ceil(base_stock - current_inventory)), 0)
            result['order'] = order
        else:
            order_uncapped = max(int(np.ceil(base_stock - current_inventory)), 0)
            cap_z = norm.ppf(0.95)
            if np.isinf(L):
                cap = empirical_mean + cap_z * empirical_std
            else:
                cap = mu_hat / (1 + L) + cap_z * sigma_hat / np.sqrt(1 + L)
            if np.isnan(cap) or np.isinf(cap):
                cap = empirical_mean + cap_z * empirical_std
            try:
                cap_int = int(np.ceil(cap))
            except (ValueError, OverflowError):
                cap = empirical_mean + cap_z * empirical_std
                cap_int = int(np.ceil(cap))
            order = max(min(order_uncapped, cap_int), 0)
            result['order'] = order
            result['order_uncapped'] = order_uncapped
            result['cap'] = cap
        return result

    def update_demand_observation(self, item_id: str, observed_demand: int):
        """
        Update observed demand history for an item.
        
        Args:
            item_id: Item identifier
            observed_demand: The true demand observed on this day (requested quantity)
        """
        self.observed_demands[item_id].append(observed_demand)

    def get_action(self, observation: str) -> tuple[str, dict]:
        """
        Generate ordering decision in JSON format with detailed statistics.
        
        Args:
            observation: Current game observation
            
        Returns:
            Tuple of (action_json_string, statistics_dict)
            statistics_dict contains all calculation details for logging
        """
        action_dict = {}
        rationale_parts = []
        statistics = {}
        for item_id in self.items_config:
            current_inventory = self._parse_inventory_from_observation(observation, item_id)
            calc_result = self._calculate_order(item_id, current_inventory)
            action_dict[item_id] = calc_result['order']
            statistics[item_id] = calc_result
            if calc_result['mu_hat'] == 0:
                rationale_parts.append(f'{item_id}: No samples yet, order=0')
            elif self.policy == 'vanilla':
                rationale_parts.append(f"{item_id}: base_stock={calc_result['base_stock']:.1f} (μ̂={calc_result['mu_hat']:.1f}, σ̂={calc_result['sigma_hat']:.1f}, z*={calc_result['z_star']:.2f}), inv={current_inventory}, order={calc_result['order']}")
            else:
                rationale_parts.append(f"{item_id}: base_stock={calc_result['base_stock']:.1f}, cap={calc_result['cap']:.1f}, inv={current_inventory}, order={calc_result['order']} (uncapped would be {calc_result['order_uncapped']})")
        policy_name = 'OR base-stock (capped)' if self.policy == 'capped' else 'OR base-stock (vanilla)'
        rationale = f'{policy_name}: ' + '; '.join(rationale_parts)
        result = {'action': action_dict, 'rationale': rationale}
        return (json.dumps(result, indent=2), statistics)

class PublishedOR(InventoryPolicy):

    def reset(self):
        config = {self.item_id: {'lead_time': self.promised_lead_time, 'profit': self.profit_per_unit, 'holding_cost': self.holding_cost_per_unit}}
        with redirect_stdout(io.StringIO()):
            self.agent = ORAgent(config, {self.item_id: self.historical_demands}, policy='capped')

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if period > 1:
            self.agent.update_demand_observation(self.item_id, previous_demand)
        self.agent.items_config[self.item_id]['profit'] = profit_per_unit
        self.agent.items_config[self.item_id]['holding_cost'] = holding_cost_per_unit
        return self.agent._calculate_order(self.item_id, int(on_hand_inventory + in_transit_total))['order']

"""Causal adaptive forecast mixture with pathwise lost-sales projection.

Our controller, not a reproduction of SabreAgent. PublishedOR remains the
stochastic branch unless a generated-validation-selected profile enables it.
No benchmark paths, IDs, future observations or shared supply sequences are
used by the controller. Existing fixed-L orders are tracked from own actions.
"""
import numpy as np
from scipy.special import gammaln

def forecasts(history, horizon):
    """Nine simple models, fitted with causal one-step residuals."""
    y = np.asarray(history, dtype=float)
    n = len(y)
    mu = []
    scales = []
    for window in (n, 4, 8, 16):
        v = y[-window:]
        mu.append(np.full(horizon, np.mean(v)))
        scales.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
    for alpha in (0.2, 0.6):
        level = y[0]
        errors = []
        for value in y[1:]:
            errors.append(value - level)
            level += alpha * (value - level)
        mu.append(np.full(horizon, level))
        scales.append(np.sqrt(np.mean(np.square(errors[-16:]))) if errors else 0.0)
    v = y[-12:]
    x = np.arange(len(v), dtype=float)
    slope = np.dot(x - x.mean(), v - v.mean()) / max(np.sum((x - x.mean()) ** 2), 1.0)
    slope *= min(1.0, max(0.0, (len(v) - 4) / 8))
    slope = np.clip(slope, -0.25 * max(1.0, v.mean()), 0.25 * max(1.0, v.mean()))
    fitted = v.mean() + slope * (x - x.mean())
    mu.append(v.mean() + slope * (len(v) - x.mean() + np.arange(horizon)))
    scales.append(np.sqrt(np.mean((v - fitted) ** 2)))
    center = y[-16:].mean()
    a = y[-17:-1] - center
    b = y[-16:] - center
    if len(a) != len(b):
        a = y[:-1] - center
        b = y[1:] - center
    phi = np.clip(np.dot(a, b) / max(np.dot(a, a), 1e-08), -0.85, 0.85)
    mu.append(center + (y[-1] - center) * phi ** (1 + np.arange(horizon)))
    scales.append(np.sqrt(np.mean((b - phi * a) ** 2)) if len(a) else 0.0)
    candidates = []
    for lag in range(2, min(14, n // 2) + 1):
        residual = y[lag:] - y[:-lag]
        candidates.append((np.mean(residual ** 2), lag))
    if candidates:
        error, lag = min(candidates)
        mu.append(np.resize(y[-lag:], horizon))
        scales.append(np.sqrt(error))
    else:
        mu.append(np.full(horizon, center))
        scales.append(np.std(y))
    points = np.maximum(0.0, np.asarray(mu))
    floor = max(1.0, 0.03 * max(1.0, np.mean(y)))
    return (points, np.maximum(scales, floor))

class AdaptiveProjection(PublishedOR):
    profile = {'discount': 0.94, 'noise': 1.0, 'stochastic': False}
    paths = 192

    def reset(self):
        super().reset()
        self.history = list(self.historical_demands)
        self.log_weights = np.zeros(9)
        for index in range(2, len(self.history)):
            f, s = forecasts(self.history[:index], 1)
            self._learn(self.history[index], f[:, 0], s)
        self.prediction = None
        self.orders = {}
        self.rng = np.random.default_rng(1042026)

    def _learn(self, value, location, scale):
        nu = 5.0
        z = (value - location) / scale
        logpdf = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(scale) - (nu + 1) / 2 * np.log1p(z * z / nu)
        self.log_weights = self.profile['discount'] * self.log_weights + logpdf
        self.log_weights -= np.max(self.log_weights)

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time == 2 and (not self.profile['stochastic']):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            if self.prediction is not None:
                self._learn(previous_demand, *self.prediction)
            self.history.append(previous_demand)
        lead = self.promised_lead_time
        f, scale = forecasts(self.history, lead + 1)
        self.prediction = (f[:, 0].copy(), scale.copy())
        weights = np.exp(self.log_weights)
        weights /= weights.sum()
        model = self.rng.choice(len(weights), size=self.paths, p=weights)
        noise = self.rng.standard_t(5, size=(self.paths, lead + 1)) * np.sqrt(3 / 5)
        samples = np.maximum(0.0, f[model] + self.profile['noise'] * scale[model, None] * noise)
        critical = profit_per_unit / (profit_per_unit + holding_cost_per_unit)
        if lead == 2:
            target = np.quantile(samples.sum(axis=1), critical)
            cap = np.quantile(samples[:, 0], 0.95)
            quantity = int(np.ceil(max(0.0, min(cap, target - on_hand_inventory - in_transit_total))))
        else:
            projected = np.full(self.paths, on_hand_inventory, dtype=float)
            for offset in range(lead):
                projected += self.orders.get(period + offset, 0.0)
                projected = np.maximum(0.0, projected - samples[:, offset])
            projected += self.orders.get(period + lead, 0.0)
            shortfall = samples[:, lead] - projected
            quantity = max(0, int(np.ceil(np.quantile(shortfall, critical))))
            self.orders[period + lead] = quantity
            self.orders.pop(period - 1, None)
        return quantity

"""Training-selected raw/log forecasts and surplus-clearance projection.

Inspired by SabreAgent Eq.3 and its two-space forecast bank, but independently
implemented: this is NOT the authors' artifact or an exact reconstruction.
Only past demand and own scheduled orders enter the deterministic controller.
The stochastic branch remains the previously frozen AdaptiveProjection.
"""
import numpy as np
from scipy.special import gammaln

def bank(history, horizon, log=False):
    y = np.asarray(history, dtype=float)
    if not log:
        return forecasts(y, horizon)
    z = np.log1p(y)
    f, s = forecasts(z * 40.0, horizon)
    return (f / 40.0, s / 40.0)

def clearance_order(shortfall, rate, profit, holding, kappa):
    """Root of empirical marginal surplus-clearance value (Sabre Eq.3)."""
    if kappa == 0:
        return max(0, int(np.ceil(np.quantile(shortfall, profit / (profit + holding)))))
    low = 0.0
    high = max(1.0, float(np.max(shortfall)) * 1.05)
    for _ in range(30):
        q = (low + high) / 2
        marginal = profit * np.mean(shortfall >= q) - holding * (np.mean(shortfall < q) + kappa * np.mean(np.maximum(0.0, q - shortfall)) / max(rate, 1.0))
        if marginal > 0:
            low = q
        else:
            high = q
    return max(0, int(np.floor(high)))

class TwoSpaceProjection(AdaptiveProjection):
    settings = dict(log=True, carry=1.5, noise=1.0, discount=0.94)
    settings_by_lead = None

    def reset(self):
        if self.settings_by_lead is not None:
            self.settings = dict(self.settings_by_lead[str(self.promised_lead_time)])
        if self.promised_lead_time == 2 or self.settings.get('original', False):
            super().reset()
            return
        self.history = list(self.historical_demands)
        self.log_weights = np.zeros(18 if self.settings['log'] else 9)
        self.orders = {}
        self.prediction = None
        self.rng = np.random.default_rng(1042027)
        for k in range(2, len(self.history)):
            self._update(self.history[k], self._forecast(self.history[:k], 1))

    def _forecast(self, history, horizon):
        f, s = bank(history, horizon)
        if self.settings['log']:
            g, v = bank(history, horizon, log=True)
            f = np.concatenate([f, g])
            s = np.concatenate([s, v])
        return (f, s)

    def _update(self, value, prediction):
        f, s = prediction
        nu = np.full(len(s), 5.0)
        observations = np.full(len(s), value)
        jacobian = np.zeros(len(s))
        if len(s) > 9:
            observations[9:] = np.log1p(value)
            jacobian[9:] = -np.log1p(value)
            nu[9:] = 14.0
        z = (observations - f[:, 0]) / s
        ll = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(s) - (nu + 1) / 2 * np.log1p(z * z / nu) + jacobian
        self.log_weights = self.settings['discount'] * self.log_weights + ll
        self.log_weights -= np.max(self.log_weights)

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time == 2 or self.settings.get('original', False):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            if self.prediction is not None:
                self._update(previous_demand, self.prediction)
            self.history.append(previous_demand)
        lead = self.promised_lead_time
        f, s = self._forecast(self.history, lead + 1)
        self.prediction = (f[:, :1].copy(), s.copy())
        weights = np.exp(self.log_weights)
        weights /= weights.sum()
        model = self.rng.choice(len(s), self.paths, p=weights)
        nu = np.where(model < 9, 5.0, 14.0)
        noise = self.rng.standard_t(nu[:, None], size=(self.paths, lead + 1)) * np.sqrt((nu[:, None] - 2) / nu[:, None])
        samples = f[model] + self.settings['noise'] * s[model, None] * noise
        if len(s) > 9:
            mask = model >= 9
            samples[mask] = np.expm1(np.minimum(samples[mask], 20.0))
        y = np.asarray(self.history, dtype=float)
        samples = np.clip(samples, 0.0, 6 * (max(y) + 3 * np.std(y) + 10))
        projected = np.full(self.paths, on_hand_inventory, dtype=float)
        for offset in range(lead):
            projected += self.orders.get(period + offset, 0.0)
            projected = np.maximum(0.0, projected - samples[:, offset])
        projected += self.orders.get(period + lead, 0.0)
        shortfall = samples[:, lead] - projected
        cv = np.clip(np.std(y) / max(np.mean(y), 1.0), 0.05, 1.5)
        carry = self.settings['carry']
        kappa = 0.0 if carry == 0 else min(12.0, 1 + carry * lead * cv)
        quantity = clearance_order(shortfall, np.mean(samples[:, lead]), profit_per_unit, holding_cost_per_unit, kappa)
        self.orders[period + lead] = quantity
        self.orders.pop(period - 1, None)
        return quantity

"""Causal capped-base-stock search with simulated iid lossy supply.

Independent compact implementation informed by SabreAgent sections 3.5/D.5.
Finite receding horizon, fixed grids and original raw demand bank differ from
the authors' full-horizon/refined search. No hidden supply sequence is used.
"""
import numpy as np

class RolloutProjection(TwoSpaceProjection):
    rollout = dict(horizon=16, paths=32, blend=True)
    parameters = np.array([(theta, cap) for theta in (1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0) for cap in (0.5, 0.8, 1.1, 1.5, 2.0, 100000.0)] + [(100000.0, cap) for cap in (0.5, 0.8, 1.1, 1.5, 2.0)])

    def reset(self):
        super().reset()
        if self.promised_lead_time == 2:
            self.survived = 3.0
            self.destroyed = 1.0
            self.last_transit = 0.0
            self.surviving_orders = {}
            self.roll_rng = np.random.default_rng(1042028)

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time != 2 or self.rollout.get('original', False):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            if self.prediction is not None:
                self._learn(previous_demand, *self.prediction)
            self.history.append(previous_demand)
            if previous_order > 0:
                lost = self.last_transit + previous_order - previous_arrivals - in_transit_total
                if lost > 0.5 * previous_order:
                    self.destroyed += 1.0
                else:
                    self.survived += 1.0
                    self.surviving_orders[period - 1] = previous_order
        self.last_transit = in_transit_total
        h = self.rollout['horizon']
        k = self.rollout['paths']
        f, s = forecasts(self.history, h + 2)
        self.prediction = (f[:, 0].copy(), s.copy())
        w = np.exp(self.log_weights)
        w /= w.sum()
        member = self.roll_rng.choice(len(w), k, p=w)
        noise = self.roll_rng.standard_t(5, size=(k, h)) * np.sqrt(3 / 5)
        demand = np.maximum(0.0, f[member, :h] + s[member, None] * noise)
        rate = np.maximum(1.0, np.convolve(w @ f, np.ones(3) / 3, mode='valid')[:h])
        survival = np.clip(self.survived / (self.survived + self.destroyed), 0.2, 1.0)
        alive = self.roll_rng.random((k, h)) < survival
        leads = self.roll_rng.integers(1, 4, size=(k, h))
        weights = np.array([self.surviving_orders.get(period - age, 0.0) * (4 - age) / 3 for age in (1, 2, 3)])
        if weights.sum() == 0:
            weights = np.array([0.5, 1 / 3, 1 / 6])
        weights /= weights.sum()
        committed = np.zeros((k, h + 3))
        for age, weight in zip((1, 2, 3), weights):
            residual = self.roll_rng.integers(0, 4 - age, size=k)
            committed[np.arange(k), residual] += in_transit_total * weight
        self.surviving_orders = {t: q for t, q in self.surviving_orders.items() if t >= period - 3}
        theta = self.parameters[:, 0, None]
        cap = self.parameters[:, 1, None]
        n = len(theta)
        stock = np.full((n, k), on_hand_inventory, dtype=float)
        transit = np.full((n, k), in_transit_total, dtype=float)
        receipts = np.broadcast_to(committed, (n, k, h + 3)).copy()
        reward = np.zeros((n, k))
        actions = None
        for step in range(h):
            order = np.floor(np.maximum(0.0, np.minimum(cap * rate[step], theta * rate[step] - stock - transit)))
            if step == 0:
                actions = order[:, 0].copy()
            finite = order * alive[None, :, step]
            transit += finite
            for delay in (1, 2, 3):
                receipts[:, :, step + delay] += finite * (leads[None, :, step] == delay)
            received = receipts[:, :, step]
            transit -= received
            stock += received
            sold = np.minimum(stock, demand[None, :, step])
            stock -= sold
            reward += profit_per_unit * sold - holding_cost_per_unit * stock
        reward -= holding_cost_per_unit * stock * stock / (2 * rate[-1])
        values = reward.mean(axis=1)
        if self.rollout['blend']:
            scale = max(1e-09, float(np.median(reward.std(axis=1, ddof=1) / np.sqrt(k))))
            blend = np.exp(np.maximum(-700.0, (values - values.max()) / scale))
            blend /= blend.sum()
            return max(0, int(np.floor(blend @ actions)))
        return int(actions[np.argmax(values)])

"""Cost-conditioned training selector over the already frozen model grid."""
import numpy as np

class CostSelectedProjection(RolloutProjection):
    settings_by_cost = None

    def reset(self):
        if self.promised_lead_time != 2:
            ratio = self.profit_per_unit / max(self.holding_cost_per_unit, 1e-12)
            cost = min((1.0, 4.0, 19.0), key=lambda value: abs(np.log(max(ratio, 1e-12) / value)))
            choices = dict(self.settings_by_lead)
            choices[str(self.promised_lead_time)] = self.settings_by_cost[f'{self.promised_lead_time}/{cost:g}']
            self.settings_by_lead = choices
        super().reset()

class InventoryBenchPolicy(CostSelectedProjection):
    profile={'discount': 0.94, 'noise': 1.0, 'stochastic': True}
    settings_by_lead={'0': {'log': False, 'carry': 0.0, 'noise': 1.0, 'discount': 0.94}, '4': {'log': False, 'carry': 0.5, 'noise': 1.0, 'discount': 0.94}, '2': {'original': True}}
    rollout={'horizon': 16, 'paths': 32, 'blend': True}
    settings_by_cost={'0/1': {'log': False, 'carry': 0.5, 'noise': 0.7, 'discount': 0.94}, '0/4': {'log': False, 'carry': 0.0, 'noise': 1.0, 'discount': 0.94}, '0/19': {'original': True}, '4/1': {'log': False, 'carry': 1.5, 'noise': 1.0, 'discount': 0.94}, '4/4': {'log': True, 'carry': 0.0, 'noise': 0.7, 'discount': 0.94}, '4/19': {'original': True}}


FrozenCostPolicy=InventoryBenchPolicy

"""Independent Sabre-inspired forecast bank, with shared level uncertainty.

Source: https://arxiv.org/html/2609.19760v2 Appendix E.1/E.2.
This is a compact reconstruction, not the authors' artifact: residual windows,
Holt initialization and seasonal fitting are explicit choices below. The parent
cost-selected zero-lead and lossy-supply branches are inherited exactly.
"""
import numpy as np
from scipy.special import gammaln

def level_bank(history, horizon):
    y = np.asarray(history, dtype=float)
    n = len(y)
    offset = np.arange(horizon)
    means = []
    sigma = []
    tau = []
    growth = []

    def add(mean, scale, parameter, g):
        means.append(np.broadcast_to(mean, (horizon,)).copy())
        sigma.append(max(1e-06, float(scale)))
        tau.append(max(0.0, float(parameter)))
        growth.append(g)
    sd = np.std(y, ddof=1) if n > 1 else 0.0
    add(y.mean(), sd, sd / np.sqrt(n), 0.0)
    for alpha in (0.15, 0.35, 0.7):
        level = y[0]
        errors = []
        for value in y[1:]:
            errors.append(value - level)
            level += alpha * (value - level)
        s = np.sqrt(np.mean(np.square(errors[-16:]))) if errors else sd
        add(level, s, s * np.sqrt(alpha / (2 - alpha)), alpha * alpha)
    level = y[0]
    trend = 0.0
    errors = []
    for value in y[1:]:
        errors.append(value - level - trend)
        old = level
        level = 0.3 * value + 0.7 * (level + trend)
        trend = 0.1 * (level - old) + 0.9 * trend
    s = np.sqrt(np.mean(np.square(errors[-16:]))) if errors else sd
    add(level + trend * (offset + 1), s, 0.6 * s, 0.25)
    for window in (4, 8):
        v = y[-window:]
        s = np.std(v, ddof=1) if len(v) > 1 else 0.0
        add(v.mean(), s, s / np.sqrt(len(v)), 0.0)
    center = y.mean()
    a = y[:-1] - center
    b = y[1:] - center
    phi = np.clip(np.dot(a, b) / max(np.dot(a, a), 1e-12), -0.95, 0.95)
    s = np.sqrt(np.mean((b - phi * a) ** 2)) if len(a) else sd
    add(center + (y[-1] - center) * phi ** (offset + 1), s, s / np.sqrt(max(n, 2)), 0.0)
    detected = max(5, n // 5 * 5)
    z = y[:min(n, detected)]
    z = z - z.mean()
    candidates = []
    for lag in range(4, 15):
        if len(z) > lag + 3:
            corr = np.dot(z[lag:], z[:-lag]) / max(np.linalg.norm(z[lag:]) * np.linalg.norm(z[:-lag]), 1e-12)
            candidates.append((corr, lag))
    lag = max(candidates)[1] if candidates else 7
    seasonal = np.array([np.mean(y[np.arange(n) % lag == (n + j) % lag]) if np.any(np.arange(n) % lag == (n + j) % lag) else y.mean() for j in offset])
    residual = y[lag:] - y[:-lag] if n > lag else y - y.mean()
    s = np.sqrt(np.mean(residual ** 2)) if len(residual) else sd
    add(seasonal, s, s / np.sqrt(n), 0.0)
    return (np.maximum(0.0, means), np.asarray(sigma), np.asarray(tau), np.asarray(growth))

class LevelUncertaintyProjection(CostSelectedProjection):
    level_settings = dict(log=True, carry=1.5, noise=1.0, discount=0.92)

    def reset(self):
        if self.promised_lead_time != 4 or self.level_settings.get('baseline', False):
            super().reset()
            return
        self.settings = dict(self.level_settings)
        self.history = list(self.historical_demands)
        self.log_weights = np.zeros(18 if self.settings['log'] else 9)
        self.orders = {}
        self.prediction = None
        self.rng = np.random.default_rng(1042051)
        for k in range(2, len(self.history)):
            self._update_level(self.history[k], self._level_forecast(self.history[:k], 1))

    def _level_forecast(self, history, horizon):
        values = level_bank(history, horizon)
        if self.settings['log']:
            other = level_bank(np.log1p(history), horizon)
            values = tuple((np.concatenate([a, b]) for a, b in zip(values, other)))
        return values

    def _update_level(self, value, prediction):
        f, s, tau, g = prediction
        scale = np.sqrt(s * s + tau * tau)
        nu = np.full(len(s), 4.0)
        observations = np.full(len(s), value)
        jacobian = np.zeros(len(s))
        if len(s) > 9:
            nu[9:] = 14.0
            observations[9:] = np.log1p(value)
            jacobian[9:] = -np.log1p(value)
        z = (observations - f[:, 0]) / scale
        ll = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(scale) - (nu + 1) / 2 * np.log1p(z * z / nu) + jacobian
        self.log_weights = self.settings['discount'] * self.log_weights + ll
        self.log_weights -= self.log_weights.max()

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time != 4 or self.level_settings.get('baseline', False):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            if self.prediction is not None:
                self._update_level(previous_demand, self.prediction)
            self.history.append(previous_demand)
        lead = self.promised_lead_time
        f, s, tau, g = self._level_forecast(self.history, lead + 1)
        self.prediction = (f[:, :1].copy(), s.copy(), tau.copy(), g.copy())
        weights = np.exp(self.log_weights)
        weights /= weights.sum()
        member = self.rng.choice(len(s), self.paths, p=weights)
        nu = np.where(member < 9, 4.0, 14.0)
        shared = self.rng.standard_t(nu) * tau[member]
        noise = self.rng.standard_t(nu[:, None], size=(self.paths, lead + 1))
        samples = f[member] + self.settings['noise'] * (shared[:, None] + s[member, None] * np.sqrt(1 + g[member, None] * np.arange(lead + 1)) * noise)
        if len(s) > 9:
            mask = member >= 9
            samples[mask] = np.expm1(np.minimum(samples[mask], 20.0))
        y = np.asarray(self.history)
        samples = np.clip(samples, 0.0, 6 * (y.max() + 3 * y.std() + 10))
        stock = np.full(self.paths, on_hand_inventory, dtype=float)
        for offset in range(lead):
            stock += self.orders.get(period + offset, 0.0)
            stock = np.maximum(0.0, stock - samples[:, offset])
        stock += self.orders.get(period + lead, 0.0)
        cv = np.clip(y.std() / max(y.mean(), 1.0), 0.05, 1.5)
        carry = self.settings['carry']
        kappa = 0.0 if carry == 0 else min(12.0, 1 + carry * lead * cv)
        order = clearance_order(samples[:, lead] - stock, np.mean(samples[:, lead]), profit_per_unit, holding_cost_per_unit, kappa)
        self.orders[period + lead] = order
        self.orders.pop(period - 1, None)
        return order

class InventoryBenchPolicy(LevelUncertaintyProjection):
    profile=FrozenCostPolicy.profile
    settings_by_lead=FrozenCostPolicy.settings_by_lead
    settings_by_cost=FrozenCostPolicy.settings_by_cost
    rollout=FrozenCostPolicy.rollout
    level_settings={'log': False, 'carry': 0.5, 'noise': 1.0, 'discount': 0.92}

FrozenLevelPolicy=InventoryBenchPolicy

"""Generated-training cost conditioning of the shared-level forecast bank."""
import numpy as np

class CostLevelProjection(LevelUncertaintyProjection):
    settings_by_ratio = None

    def reset(self):
        if self.promised_lead_time == 4:
            ratio = self.profit_per_unit / max(self.holding_cost_per_unit, 1e-12)
            nearest = min((1.0, 4.0, 19.0), key=lambda v: abs(np.log(max(ratio, 1e-12) / v)))
            self.level_settings = dict(self.settings_by_ratio[f'{nearest:g}'])
        super().reset()

class InventoryBenchPolicy(CostLevelProjection):
    profile=FrozenLevelPolicy.profile
    settings_by_lead=FrozenLevelPolicy.settings_by_lead
    settings_by_cost=FrozenLevelPolicy.settings_by_cost
    rollout=FrozenLevelPolicy.rollout
    level_settings=FrozenLevelPolicy.level_settings
    settings_by_ratio={'1': {'log': True, 'carry': 1.5, 'noise': 1.0, 'discount': 0.85}, '4': {'log': False, 'carry': 1.5, 'noise': 0.7, 'discount': 0.92}, '19': {'log': False, 'carry': 0.5, 'noise': 0.7, 'discount': 0.92}}

FrozenLevelCostPolicy=InventoryBenchPolicy

"""Causal regression and harmonic extensions to the frozen statistical bank.

No calendar, product identity, official demand or hidden supply enters decisions.
The parent zero-lead and stochastic branches are inherited without changes.
"""
import numpy as np
from scipy.special import gammaln

def extended_bank(history, horizon):
    y = np.asarray(history, dtype=float)
    n = len(y)
    future = np.arange(1, horizon + 1)
    f, s, tau, g = level_bank(y, horizon)
    means = list(f)
    scales = list(s)
    parameters = list(tau)
    growth = list(g)
    for window in (8, 16):
        v = y[-window:]
        m = len(v)
        x = np.arange(m) - (m - 1) / 2
        slope = np.dot(x, v) / max(np.dot(x, x), 1.0)
        slope *= min(1.0, max(0.0, (m - 4) / 8))
        slope = np.clip(slope, -0.2 * max(v.mean(), 1.0), 0.2 * max(v.mean(), 1.0))
        residual = v - v.mean() - slope * x
        sd = max(1e-06, np.sqrt(np.mean(residual ** 2)))
        means.append(v.mean() + slope * ((m - 1) / 2 + future))
        scales.append(sd)
        parameters.append(sd / np.sqrt(m))
        growth.append(1 / max(np.dot(x, x), 1.0))
    v = y[-32:]
    m = len(v)
    x = np.arange(m)
    best = None
    for period in range(4, min(14, m // 2) + 1):
        centered = (x - x.mean()) / max(m, 1)
        design = np.column_stack((np.ones(m), centered, np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)))
        penalty = np.diag([1e-08, 4.0, 2.0, 2.0])
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ v)
        residual = v - design @ coef
        error = np.mean(residual ** 2)
        criterion = error * (1 + 4 / max(m - 4, 1))
        if best is None or criterion < best[0]:
            best = (criterion, period, coef, error)
    if best is None:
        forecast = np.full(horizon, v.mean())
        sd = max(1e-06, v.std())
    else:
        _, period, coef, error = best
        xf = m - 1 + future
        design = np.column_stack((np.ones(horizon), (xf - x.mean()) / m, np.sin(2 * np.pi * xf / period), np.cos(2 * np.pi * xf / period)))
        forecast = design @ coef
        sd = max(1e-06, np.sqrt(error))
    means.append(forecast)
    scales.append(sd)
    parameters.append(2 * sd / np.sqrt(m))
    growth.append(0.05)
    return (np.maximum(0.0, means), np.asarray(scales), np.asarray(parameters), np.asarray(growth))

class TrendBankProjection(CostLevelProjection):
    extension = dict(baseline=True)
    raw_models = 12

    def reset(self):
        extension = self.extension
        self.extension = dict(baseline=True)
        super().reset()
        self.extension = extension
        if self.promised_lead_time != 4 or self.extension.get('baseline', False):
            return
        self.log_weights = np.zeros(self.raw_models * (2 if self.settings['log'] else 1))
        self.history = list(self.historical_demands)
        self.prediction = None
        self.orders = {}
        self.rng = np.random.default_rng(1042051)
        for k in range(2, len(self.history)):
            self._update_level(self.history[k], self._level_forecast(self.history[:k], 1))

    def _level_forecast(self, history, horizon):
        if self.extension.get('baseline', False):
            return super()._level_forecast(history, horizon)
        values = extended_bank(history, horizon)
        if self.settings['log']:
            other = extended_bank(np.log1p(history), horizon)
            values = tuple((np.concatenate([a, b]) for a, b in zip(values, other)))
        return values

    def _update_level(self, value, prediction):
        if self.extension.get('baseline', False):
            return super()._update_level(value, prediction)
        f, s, tau, g = prediction
        scale = np.sqrt(s * s + tau * tau)
        nu = np.full(len(s), 4.0)
        observations = np.full(len(s), value)
        jacobian = np.zeros(len(s))
        if len(s) > self.raw_models:
            nu[self.raw_models:] = 14.0
            observations[self.raw_models:] = np.log1p(value)
            jacobian[self.raw_models:] = -np.log1p(value)
        z = (observations - f[:, 0]) / scale
        ll = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(scale) - (nu + 1) / 2 * np.log1p(z * z / nu) + jacobian
        self.log_weights = self.settings['discount'] * self.log_weights + ll
        self.log_weights -= self.log_weights.max()

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time != 4 or self.extension.get('baseline', False):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            if self.prediction is not None:
                self._update_level(previous_demand, self.prediction)
            self.history.append(previous_demand)
        lead = self.promised_lead_time
        f, s, tau, g = self._level_forecast(self.history, lead + 1)
        self.prediction = (f[:, :1].copy(), s.copy(), tau.copy(), g.copy())
        weights = np.exp(self.log_weights)
        weights /= weights.sum()
        member = self.rng.choice(len(s), self.paths, p=weights)
        nu = np.where(member < self.raw_models, 4.0, 14.0)
        shared = self.rng.standard_t(nu) * tau[member]
        noise = self.rng.standard_t(nu[:, None], size=(self.paths, lead + 1))
        multiplier = self.settings['noise'] * self.extension['noise']
        samples = f[member] + multiplier * (shared[:, None] + s[member, None] * np.sqrt(1 + g[member, None] * np.arange(lead + 1)) * noise)
        if len(s) > self.raw_models:
            mask = member >= self.raw_models
            samples[mask] = np.expm1(np.minimum(samples[mask], 20.0))
        y = np.asarray(self.history)
        samples = np.clip(samples, 0.0, 6 * (y.max() + 3 * y.std() + 10))
        stock = np.full(self.paths, on_hand_inventory, dtype=float)
        for offset in range(lead):
            stock += self.orders.get(period + offset, 0.0)
            stock = np.maximum(0.0, stock - samples[:, offset])
        stock += self.orders.get(period + lead, 0.0)
        cv = np.clip(y.std() / max(y.mean(), 1.0), 0.05, 1.5)
        carry = self.settings['carry'] * self.extension['carry']
        kappa = 0.0 if carry == 0 else min(12.0, 1 + carry * lead * cv)
        order = clearance_order(samples[:, lead] - stock, np.mean(samples[:, lead]), profit_per_unit, holding_cost_per_unit, kappa)
        self.orders[period + lead] = order
        self.orders.pop(period - 1, None)
        return order

"""Generated-training richer supply forecasts and denser capped-stock grid."""
import numpy as np
from scipy.special import gammaln

class SupplyBankProjection(CostLevelProjection):
    supply_extension = dict(baseline=True)
    refined_parameters = np.array([(theta, cap) for theta in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0) for cap in (0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.5, 2.0, 100000.0)] + [(100000.0, cap) for cap in (0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.5, 2.0)])

    def reset(self):
        setting = self.supply_extension
        self.supply_extension = dict(baseline=True)
        super().reset()
        self.supply_extension = setting
        if self.promised_lead_time != 2 or self.supply_extension.get('baseline', False):
            return
        self.log_weights = np.zeros(12 if self.supply_extension['bank'] else 9)
        self.history = list(self.historical_demands)
        self.prediction = None
        for k in range(2, len(self.history)):
            f, s = self._supply_forecast(self.history[:k], 1)
            self._learn(self.history[k], f[:, 0], s)

    def _supply_forecast(self, history, horizon):
        if not self.supply_extension.get('bank', False):
            return forecasts(history, horizon)
        f, s, tau, g = extended_bank(history, horizon)
        return (f, np.sqrt(s * s + tau * tau))

    def _learn(self, value, location, scale):
        if self.promised_lead_time != 2 or self.supply_extension.get('baseline', False):
            return super()._learn(value, location, scale)
        nu = 5.0
        z = (value - location) / scale
        ll = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(scale) - (nu + 1) / 2 * np.log1p(z * z / nu)
        self.log_weights = self.profile['discount'] * self.log_weights + ll
        self.log_weights -= self.log_weights.max()

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time != 2 or self.supply_extension.get('baseline', False):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            if self.prediction is not None:
                self._learn(previous_demand, *self.prediction)
            self.history.append(previous_demand)
            if previous_order > 0:
                lost = self.last_transit + previous_order - previous_arrivals - in_transit_total
                if lost > 0.5 * previous_order:
                    self.destroyed += 1.0
                else:
                    self.survived += 1.0
                    self.surviving_orders[period - 1] = previous_order
        self.last_transit = in_transit_total
        h = self.rollout['horizon']
        k = self.rollout['paths']
        f, s = self._supply_forecast(self.history, h + 2)
        self.prediction = (f[:, 0].copy(), s.copy())
        w = np.exp(self.log_weights)
        w /= w.sum()
        member = self.roll_rng.choice(len(w), k, p=w)
        noise = self.roll_rng.standard_t(5, size=(k, h)) * np.sqrt(3 / 5)
        demand = np.maximum(0.0, f[member, :h] + s[member, None] * noise)
        rate = np.maximum(1.0, np.convolve(w @ f, np.ones(3) / 3, mode='valid')[:h])
        survival = np.clip(self.survived / (self.survived + self.destroyed), 0.2, 1.0)
        alive = self.roll_rng.random((k, h)) < survival
        leads = self.roll_rng.integers(1, 4, size=(k, h))
        weights = np.array([self.surviving_orders.get(period - age, 0.0) * (4 - age) / 3 for age in (1, 2, 3)])
        if weights.sum() == 0:
            weights = np.array([0.5, 1 / 3, 1 / 6])
        weights /= weights.sum()
        committed = np.zeros((k, h + 3))
        for age, weight in zip((1, 2, 3), weights):
            residual = self.roll_rng.integers(0, 4 - age, size=k)
            committed[np.arange(k), residual] += in_transit_total * weight
        self.surviving_orders = {t: q for t, q in self.surviving_orders.items() if t >= period - 3}
        parameters = self.refined_parameters if self.supply_extension['refine'] else self.parameters
        theta = parameters[:, 0, None]
        cap = parameters[:, 1, None]
        n = len(theta)
        stock = np.full((n, k), on_hand_inventory, dtype=float)
        transit = np.full((n, k), in_transit_total, dtype=float)
        receipts = np.broadcast_to(committed, (n, k, h + 3)).copy()
        reward = np.zeros((n, k))
        actions = None
        for step in range(h):
            order = np.floor(np.maximum(0.0, np.minimum(cap * rate[step], theta * rate[step] - stock - transit)))
            if step == 0:
                actions = order[:, 0].copy()
            finite = order * alive[None, :, step]
            transit += finite
            for delay in (1, 2, 3):
                receipts[:, :, step + delay] += finite * (leads[None, :, step] == delay)
            received = receipts[:, :, step]
            transit -= received
            stock += received
            sold = np.minimum(stock, demand[None, :, step])
            stock -= sold
            reward += profit_per_unit * sold - holding_cost_per_unit * stock
        reward -= holding_cost_per_unit * stock * stock / (2 * rate[-1])
        values = reward.mean(axis=1)
        if self.rollout['blend']:
            scale = max(1e-09, float(np.median(reward.std(axis=1, ddof=1) / np.sqrt(k))))
            blend = np.exp(np.maximum(-700.0, (values - values.max()) / scale))
            blend /= blend.sum()
            return max(0, int(np.floor(blend @ actions)))
        return int(actions[np.argmax(values)])

class InventoryBenchPolicy(SupplyBankProjection):
    profile=FrozenLevelCostPolicy.profile
    settings_by_lead=FrozenLevelCostPolicy.settings_by_lead
    settings_by_cost=FrozenLevelCostPolicy.settings_by_cost
    rollout=FrozenLevelCostPolicy.rollout
    level_settings=FrozenLevelCostPolicy.level_settings
    settings_by_ratio=FrozenLevelCostPolicy.settings_by_ratio
    supply_extension={'bank': True, 'refine': True}

FrozenSupplyBankPolicy=InventoryBenchPolicy

"""Frozen category/calendar priors with causal, likelihood-weighted variants.

Independent implementation inspired by SabreAgent sections D.3/E.3. No SKU
tables, year-specific benchmark events, demand lookups, or episode lengths.
Unrecognized text/dates and other supply regimes inherit the exact parent.
"""
from datetime import datetime, timedelta
import re
import numpy as np
from scipy.special import gammaln, logsumexp

def parse_date(value):
    try:
        return datetime.strptime(str(value).replace('/', '-'), '%Y-%m-%d')
    except (ValueError, TypeError):
        return None

def category_prior(text):
    text = str(text or '').lower()
    summer = '\\b(swim\\w*|bikini\\w*|beach\\w*|shorts|sandal\\w*|linen|camisole\\w*|vest top|strap top|tank top|short.sleeved)\\b'
    winter = '\\b(coat\\w*|jacket\\w*|parka\\w*|puffer\\w*|sweater\\w*|jumper\\w*|cardigan\\w*|hoodie\\w*|fleece|boot\\w*|tights|long.sleeved)\\b'
    if re.search(summer, text):
        return 1.0
    if re.search(winter, text):
        return -1.0
    return 0.0

def annual_shape(dates, direction, amplitude):
    phase = np.array([(d.timetuple().tm_yday - 1) / 365.25 for d in dates])
    grid = np.arange(366) / 365.25
    normalization = np.mean(np.exp(amplitude * direction * np.cos(2 * np.pi * (grid - 0.5))))
    return np.exp(amplitude * direction * np.cos(2 * np.pi * (phase - 0.5))) / normalization

class CalendarProjection(SupplyBankProjection):
    calendar_setting = dict(baseline=True)

    def reset(self):
        super().reset()
        self.calendar_active = False
        if self.promised_lead_time != 4 or self.calendar_setting.get('baseline', False):
            return
        dates = [parse_date(d) for d, _ in self.initial_samples]
        direction = category_prior(self.product_description)
        if not direction or not all(dates):
            return
        deltas = [(b - a).days for a, b in zip(dates, dates[1:])]
        if not deltas or min(deltas) <= 0:
            return
        self.calendar_step = int(np.median(deltas))
        self.calendar_dates = dates
        self.calendar_direction = direction
        self.calendar_amplitudes = np.array([0.0, 0.5, 1.0]) * self.calendar_setting['amplitude']
        self.calendar_active = True
        members = 18 if self.settings['log'] else 9
        self.shape_log_weights = np.zeros((3, members))
        prior = self.calendar_setting['prior']
        self.variant_log_weights = np.log([1 - prior, prior / 2, prior / 2])
        for k in range(2, len(self.history)):
            prediction = self._shape_forecast(self.history[:k], dates[:k], [dates[k]])
            self._shape_learn(self.history[k], prediction)
        self.prediction = None
        self.last_calendar_date = None

    def _shape_forecast(self, history, dates, future):
        predictions = []
        multipliers = []
        for amplitude in self.calendar_amplitudes:
            past = annual_shape(dates, self.calendar_direction, amplitude)
            shape = annual_shape(future, self.calendar_direction, amplitude)
            y = np.asarray(history) / past
            values = level_bank(y, len(future))
            if self.settings['log']:
                other = level_bank(np.log1p(y), len(future))
                values = tuple((np.concatenate([a, b]) for a, b in zip(values, other)))
            predictions.append(values)
            multipliers.append(shape)
        return tuple((np.stack([v[j] for v in predictions]) for j in range(4))) + (np.array(multipliers),)

    def _shape_learn(self, value, prediction):
        f, s, tau, g, shape = prediction
        scale = np.sqrt(s * s + tau * tau)
        observations = np.broadcast_to(value / shape[:, :1], s.shape).copy()
        nu = np.full(s.shape, 4.0)
        jacobian = np.broadcast_to(-np.log(shape[:, :1]), s.shape).copy()
        if s.shape[1] > 9:
            nu[:, 9:] = 14.0
            jacobian[:, 9:] -= np.log1p(observations[:, 9:])
            observations[:, 9:] = np.log1p(observations[:, 9:])
        z = (observations - f[:, :, 0]) / scale
        ll = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(scale) - (nu + 1) / 2 * np.log1p(z * z / nu) + jacobian
        member_log = self.shape_log_weights - logsumexp(self.shape_log_weights, axis=1)[:, None]
        variant_ll = logsumexp(member_log + ll, axis=1)
        self.variant_log_weights = self.settings['discount'] * self.variant_log_weights + variant_ll
        self.variant_log_weights -= self.variant_log_weights.max()
        self.shape_log_weights = self.settings['discount'] * self.shape_log_weights + ll
        self.shape_log_weights -= self.shape_log_weights.max(axis=1)[:, None]

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if not self.calendar_active:
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        date = parse_date(current_date)
        if date is None:
            date = (self.last_calendar_date or self.calendar_dates[-1]) + timedelta(days=self.calendar_step)
        if period > 1:
            if self.prediction is not None:
                self._shape_learn(previous_demand, self.prediction)
            self.history.append(previous_demand)
            self.calendar_dates.append(self.last_calendar_date)
        self.last_calendar_date = date
        lead = self.promised_lead_time
        future = [date + timedelta(days=self.calendar_step * k) for k in range(lead + 1)]
        f, s, tau, g, shape = self._shape_forecast(self.history, self.calendar_dates, future)
        self.prediction = (f[:, :, :1].copy(), s.copy(), tau.copy(), g.copy(), shape[:, :1].copy())
        vw = np.exp(self.variant_log_weights)
        vw /= vw.sum()
        variant = self.rng.choice(3, self.paths, p=vw)
        mw = np.exp(self.shape_log_weights)
        mw /= mw.sum(axis=1)[:, None]
        member = (self.rng.random(self.paths)[:, None] > np.cumsum(mw[variant], axis=1)).sum(axis=1)
        member = np.minimum(member, s.shape[1] - 1)
        nu = np.where(member < 9, 4.0, 14.0)
        shared = self.rng.standard_t(nu) * tau[variant, member]
        noise = self.rng.standard_t(nu[:, None], size=(self.paths, lead + 1))
        samples = f[variant, member] + self.settings['noise'] * (shared[:, None] + s[variant, member, None] * np.sqrt(1 + g[variant, member, None] * np.arange(lead + 1)) * noise)
        mask = member >= 9
        samples[mask] = np.expm1(np.minimum(samples[mask], 20.0))
        samples *= shape[variant]
        y = np.asarray(self.history)
        samples = np.clip(samples, 0.0, 6 * (y.max() + 3 * y.std() + 10))
        stock = np.full(self.paths, on_hand_inventory, dtype=float)
        for offset in range(lead):
            stock += self.orders.get(period + offset, 0.0)
            stock = np.maximum(0.0, stock - samples[:, offset])
        stock += self.orders.get(period + lead, 0.0)
        cv = np.clip(y.std() / max(y.mean(), 1.0), 0.05, 1.5)
        carry = self.settings['carry']
        kappa = 0.0 if carry == 0 else min(12.0, 1 + carry * lead * cv)
        order = clearance_order(samples[:, lead] - stock, samples[:, lead].mean(), profit_per_unit, holding_cost_per_unit, kappa)
        self.orders[period + lead] = order
        self.orders.pop(period - 1, None)
        return order

class InventoryBenchPolicy(CalendarProjection):
    profile=FrozenSupplyBankPolicy.profile
    settings_by_lead=FrozenSupplyBankPolicy.settings_by_lead
    settings_by_cost=FrozenSupplyBankPolicy.settings_by_cost
    rollout=FrozenSupplyBankPolicy.rollout
    level_settings=FrozenSupplyBankPolicy.level_settings
    settings_by_ratio=FrozenSupplyBankPolicy.settings_by_ratio
    supply_extension=FrozenSupplyBankPolicy.supply_extension
    calendar_setting={'amplitude': 0.8, 'prior': 0.5}

FrozenCalendarPolicy=InventoryBenchPolicy

"""Generated-training calibration of innovation and shared-level uncertainty."""
import numpy as np

class CalibratedProjection(CalendarProjection):
    calibration = dict(baseline=True)

    def _calibrate(self, values):
        if self.calibration.get('baseline', False):
            return values
        f, s, tau, g = values
        s = s * self.calibration['innovation']
        tau = tau * self.calibration['shared']
        floor = self.calibration.get('floor', 0.0)
        if floor:
            minimum = floor * np.maximum(1.0, np.abs(f[..., 0]))
            if s.shape[-1] > 9:
                minimum[..., 9:] = floor
            s = np.maximum(s, minimum)
        return (f, s, tau, g)

    def _level_forecast(self, history, horizon):
        values = super()._level_forecast(history, horizon)
        if self.promised_lead_time != 4:
            return values
        return self._calibrate(values)

    def _shape_forecast(self, history, dates, future):
        *values, shape = super()._shape_forecast(history, dates, future)
        return self._calibrate(tuple(values)) + (shape,)

class InventoryBenchPolicy(CalibratedProjection):
    profile=FrozenCalendarPolicy.profile
    settings_by_lead=FrozenCalendarPolicy.settings_by_lead
    settings_by_cost=FrozenCalendarPolicy.settings_by_cost
    rollout=FrozenCalendarPolicy.rollout
    level_settings=FrozenCalendarPolicy.level_settings
    settings_by_ratio=FrozenCalendarPolicy.settings_by_ratio
    supply_extension=FrozenCalendarPolicy.supply_extension
    calendar_setting=FrozenCalendarPolicy.calendar_setting
    calibration={'innovation': 1.1, 'shared': 1.0, 'floor': 0.0}

FrozenCalibrationPolicy=InventoryBenchPolicy

"""Causal seasonal peak uncertainty over the frozen calibrated controller."""
from datetime import timedelta
import numpy as np

def phase_shape(dates, direction, amplitude, shift):
    phase = np.array([(d.timetuple().tm_yday - 1) / 365.25 for d in dates])
    grid = np.arange(366) / 365.25
    normalizer = np.mean(np.exp(amplitude * direction * np.cos(2 * np.pi * (grid - 0.5 - shift))))
    return np.exp(amplitude * direction * np.cos(2 * np.pi * (phase - 0.5 - shift))) / normalizer

class PhaseProjection(CalibratedProjection):
    phase_setting = dict(baseline=True)

    def reset(self):
        self.phase_live = False
        super().reset()
        if not self.calendar_active or self.phase_setting.get('baseline', False):
            return
        self.phase_live = True
        delta = self.phase_setting['shift']
        mass = self.phase_setting['mass']
        self.phase_variants = [(0.0, 0.0)] + [(a, s) for a in self.calendar_amplitudes[1:] for s in (0.0, -delta, delta)]
        amplitude_mass = np.array([1 - self.calendar_setting['prior'], self.calendar_setting['prior'] / 2, self.calendar_setting['prior'] / 2])
        weights = np.r_[amplitude_mass[0], np.outer(amplitude_mass[1:], [1 - mass, mass / 2, mass / 2]).ravel()]
        self.variant_log_weights = np.log(np.maximum(weights, 1e-300))
        self.shape_log_weights = np.zeros((len(weights), 18 if self.settings['log'] else 9))
        for k in range(2, len(self.history)):
            self._shape_learn(self.history[k], self._shape_forecast(self.history[:k], self.calendar_dates[:k], [self.calendar_dates[k]]))
        self.prediction = None

    def _shape_forecast(self, history, dates, future):
        if not self.phase_live:
            return super()._shape_forecast(history, dates, future)
        predictions = []
        multipliers = []
        for amplitude, shift in self.phase_variants:
            past = phase_shape(dates, self.calendar_direction, amplitude, shift)
            shape = phase_shape(future, self.calendar_direction, amplitude, shift)
            y = np.asarray(history) / past
            values = level_bank(y, len(future))
            if self.settings['log']:
                other = level_bank(np.log1p(y), len(future))
                values = tuple((np.concatenate([a, b]) for a, b in zip(values, other)))
            predictions.append(values)
            multipliers.append(shape)
        values = tuple((np.stack([v[j] for v in predictions]) for j in range(4)))
        return self._calibrate(values) + (np.array(multipliers),)

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if not self.phase_live:
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        date = parse_date(current_date)
        if date is None:
            date = (self.last_calendar_date or self.calendar_dates[-1]) + timedelta(days=self.calendar_step)
        if period > 1:
            if self.prediction is not None:
                self._shape_learn(previous_demand, self.prediction)
            self.history.append(previous_demand)
            self.calendar_dates.append(self.last_calendar_date)
        self.last_calendar_date = date
        lead = self.promised_lead_time
        future = [date + timedelta(days=self.calendar_step * k) for k in range(lead + 1)]
        f, s, tau, g, shape = self._shape_forecast(self.history, self.calendar_dates, future)
        self.prediction = (f[:, :, :1].copy(), s.copy(), tau.copy(), g.copy(), shape[:, :1].copy())
        vw = np.exp(self.variant_log_weights)
        vw /= vw.sum()
        variant = self.rng.choice(len(vw), self.paths, p=vw)
        mw = np.exp(self.shape_log_weights)
        mw /= mw.sum(axis=1)[:, None]
        member = (self.rng.random(self.paths)[:, None] > np.cumsum(mw[variant], axis=1)).sum(axis=1)
        member = np.minimum(member, s.shape[1] - 1)
        nu = np.where(member < 9, 4.0, 14.0)
        shared = self.rng.standard_t(nu) * tau[variant, member]
        noise = self.rng.standard_t(nu[:, None], size=(self.paths, lead + 1))
        samples = f[variant, member] + self.settings['noise'] * (shared[:, None] + s[variant, member, None] * np.sqrt(1 + g[variant, member, None] * np.arange(lead + 1)) * noise)
        mask = member >= 9
        samples[mask] = np.expm1(np.minimum(samples[mask], 20.0))
        samples *= shape[variant]
        y = np.asarray(self.history)
        samples = np.clip(samples, 0.0, 6 * (y.max() + 3 * y.std() + 10))
        stock = np.full(self.paths, on_hand_inventory, dtype=float)
        for offset in range(lead):
            stock += self.orders.get(period + offset, 0.0)
            stock = np.maximum(0.0, stock - samples[:, offset])
        stock += self.orders.get(period + lead, 0.0)
        cv = np.clip(y.std() / max(y.mean(), 1.0), 0.05, 1.5)
        carry = self.settings['carry']
        kappa = 0.0 if carry == 0 else min(12.0, 1 + carry * lead * cv)
        order = clearance_order(samples[:, lead] - stock, samples[:, lead].mean(), profit_per_unit, holding_cost_per_unit, kappa)
        self.orders[period + lead] = order
        self.orders.pop(period - 1, None)
        return order

class InventoryBenchPolicy(PhaseProjection):
    profile=FrozenCalibrationPolicy.profile
    settings_by_lead=FrozenCalibrationPolicy.settings_by_lead
    settings_by_cost=FrozenCalibrationPolicy.settings_by_cost
    rollout=FrozenCalibrationPolicy.rollout
    level_settings=FrozenCalibrationPolicy.level_settings
    settings_by_ratio=FrozenCalibrationPolicy.settings_by_ratio
    supply_extension=FrozenCalibrationPolicy.supply_extension
    calendar_setting=FrozenCalibrationPolicy.calendar_setting
    calibration=FrozenCalibrationPolicy.calibration
    phase_setting={'shift': 0.04, 'mass': 0.5}

FrozenPhasePolicy=InventoryBenchPolicy

"""Deterministic component quadrature for the causal zero-lead mixture.

Every forecast member receives its exact online posterior weight. Student-t
quantile nodes approximate each component; no random member selection enters.
All positive-lead branches inherit the frozen PhaseProjection unchanged.
"""
from functools import lru_cache
import numpy as np
from scipy.special import stdtrit

@lru_cache(None)
def nodes(n, df):
    return stdtrit(df, (np.arange(n) + 0.5) / n) * np.sqrt((df - 2) / df)

def weighted_order(demand, weights, stock, profit, holding, carry):
    indices = np.argsort(demand, kind='stable')
    d = demand[indices]
    w = weights[indices]
    w = w / w.sum()
    cw = np.cumsum(w)
    cw[-1] = 1.0
    cd = np.cumsum(w * d)
    if carry == 0:
        target = d[min(np.searchsorted(cw, profit / (profit + holding)), len(d) - 1)]
        return max(0, int(np.ceil(target - stock)))
    rate = max(float(cd[-1]), 1.0)
    k = holding * carry / rate
    right = profit - (profit + holding) * cw - k * (d * cw - cd)
    j = min(int(np.searchsorted(-right, 0.0)), len(d) - 1)
    if j == 0:
        target = d[0]
    else:
        target = (profit - (profit + holding) * cw[j - 1] + k * cd[j - 1]) / (k * cw[j - 1])
        target = min(float(d[j]), max(float(d[j - 1]), float(target)))
    return max(0, int(np.floor(target - stock + 1e-09)))

class MixtureProjection(PhaseProjection):
    mixture_setting = dict(baseline=True)

    def reset(self):
        super().reset()
        if self.promised_lead_time != 0 or self.mixture_setting.get('baseline', False):
            return
        inherited = dict(self.settings)
        original = inherited.get('original', False)
        self.settings = dict(log=self.mixture_setting['log'], discount=self.profile['discount'] if original else inherited['discount'], noise=self.profile['noise'] if original else inherited['noise'], carry=0.0 if original or inherited.get('carry', 0.0) == 0 else 1.0)
        self.settings['noise'] *= self.mixture_setting['noise']
        if 'carry' in self.mixture_setting:
            self.settings['carry'] = self.mixture_setting['carry']
        self.history = list(self.historical_demands)
        self.prediction = None
        self.orders = {}
        self.log_weights = np.zeros(18 if self.settings['log'] else 9)
        for k in range(2, len(self.history)):
            TwoSpaceProjection._update(self, self.history[k], TwoSpaceProjection._forecast(self, self.history[:k], 1))

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time != 0 or self.mixture_setting.get('baseline', False):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            if self.prediction is not None:
                TwoSpaceProjection._update(self, previous_demand, self.prediction)
            self.history.append(previous_demand)
        f, s = TwoSpaceProjection._forecast(self, self.history, 1)
        self.prediction = (f.copy(), s.copy())
        w = np.exp(self.log_weights)
        w /= w.sum()
        n = self.mixture_setting.get('nodes', 128)
        z = np.tile(nodes(n, 5.0), (len(s), 1))
        if len(s) > 9:
            z[9:] = nodes(n, 14.0)
        d = f + self.settings['noise'] * s[:, None] * z
        if len(s) > 9:
            d[9:] = np.expm1(np.minimum(d[9:], 20.0))
        y = np.asarray(self.history)
        d = np.clip(d, 0.0, 6 * (y.max() + 3 * y.std() + 10))
        weights = np.broadcast_to(w[:, None] / n, d.shape).ravel()
        q = weighted_order(d.ravel(), weights, on_hand_inventory, profit_per_unit, holding_cost_per_unit, self.settings['carry'])
        self.orders[period] = q
        self.orders.pop(period - 1, None)
        return q

"""Training-selected low-margin mixture; exact raw parent at other margins."""
import sys
from pathlib import Path

class LowCostMixture(MixtureProjection):
    cost_setting = dict(baseline=True)

    def reset(self):
        low = self.profit_per_unit / max(self.holding_cost_per_unit, 1e-12) <= 2.0
        self.mixture_setting = dict(log=False, noise=1.0, nodes=128)
        if self.promised_lead_time == 0 and low and (not self.cost_setting.get('baseline', False)):
            self.mixture_setting = dict(self.cost_setting)
        super().reset()

class InventoryBenchPolicy(LowCostMixture):
    profile=FrozenPhasePolicy.profile
    settings_by_lead=FrozenPhasePolicy.settings_by_lead
    settings_by_cost=FrozenPhasePolicy.settings_by_cost
    rollout=FrozenPhasePolicy.rollout
    level_settings=FrozenPhasePolicy.level_settings
    settings_by_ratio=FrozenPhasePolicy.settings_by_ratio
    supply_extension=FrozenPhasePolicy.supply_extension
    calendar_setting=FrozenPhasePolicy.calendar_setting
    calibration=FrozenPhasePolicy.calibration
    phase_setting=FrozenPhasePolicy.phase_setting
    mixture_setting=dict(log=False,noise=1.,nodes=128)
    cost_setting={'log': True, 'noise': 1.15, 'nodes': 128}

FrozenPolicy=InventoryBenchPolicy
from scipy.stats import qmc
from scipy.special import stdtrit
class SobolRNG:
    """One aligned power-of-two block per forecast, no point thinning.

    Choice starts a block. Columns cover variant, member, shared innovation,
    then five individual innovations. Unseasonal forecasts consume seven
    columns. Each block continues the same scrambled infinite sequence;
    the engine knows neither horizon remaining nor any identifier or demand.
    """
    def __init__(self, paths):
        if paths <= 0 or paths & (paths - 1):
            raise ValueError('Sobol path count must be a power of two')
        self.paths = paths
        self.engine = qmc.Sobol(d=8, scramble=True, seed=1042051)
        self.block = None
        self.column = 0

    def _uniform(self, size):
        shape = (size,) if np.isscalar(size) else tuple(size)
        if shape[0] != self.paths or len(shape) > 2:
            raise ValueError('Unexpected forecast random shape')
        width = 1 if len(shape) == 1 else shape[1]
        if self.block is None or self.column + width > 8:
            raise ValueError('Predictive Sobol dimensions exhausted')
        values = self.block[:, self.column:self.column + width]
        self.column += width
        return values[:, 0] if len(shape) == 1 else values

    def choice(self, count, size, p):
        if size != self.paths:
            raise ValueError('Unexpected mixture sample count')
        self.block = self.engine.random(self.paths)
        self.column = 0
        cumulative = np.cumsum(p)
        cumulative[-1] = 1.
        return np.minimum(np.searchsorted(cumulative, self._uniform(size)), count-1)

    def random(self, size):
        return self._uniform(size)

    def standard_t(self, df, size=None):
        if size is None:
            size = np.shape(df)
        uniforms = np.clip(self._uniform(size), 1e-12, 1-1e-12)
        return stdtrit(df, uniforms)


class QMCProjection(FrozenPolicy):
    precision_setting = dict(baseline=True)

    def reset(self):
        self.paths = 192
        super().reset()
        if self.promised_lead_time != 4 or self.precision_setting.get('baseline', False):
            return
        self.paths = self.precision_setting['paths']
        if self.precision_setting['method'] == 'sobol':
            self.rng = SobolRNG(self.paths)

class InventoryBenchPolicy(QMCProjection):
    precision_setting={'method': 'sobol', 'paths': 1024}

ParentPolicy=InventoryBenchPolicy

"""Exact belief over outstanding finite-lead receipts from public observations.

Supply assumption: independently lost orders or a uniform finite lead in 1..3,
matching the documented benchmark. Actual lead times are never observed.
Only the stochastic branch changes; all forecast and rollout coefficients stay
at the frozen parent. Parent random draws are consumed identically.
"""
import numpy as np

class ReceiptBelief:

    def __init__(self):
        self.states = {(0.0, 0.0, 0.0): 1.0}
        self.previous_transit = 0.0

    def observe(self, previous_order, previous_arrivals, transit):
        """Condition old receipts, shift one period, add the last surviving order."""
        q = float(previous_order)
        survived = float(transit) - self.previous_transit + float(previous_arrivals)
        if abs(survived) < 1e-07:
            survived = 0.0
        elif abs(survived - q) < 1e-07:
            survived = q
        else:
            raise ValueError('Supply observation incompatible with whole-order loss')
        updated = {}
        for state, probability in self.states.items():
            if abs(state[0] - previous_arrivals) > 1e-07:
                continue
            shifted = np.array([state[1], state[2], 0.0])
            for delay in (0, 1, 2):
                next_state = shifted.copy()
                next_state[delay] += survived
                key = tuple(next_state)
                updated[key] = updated.get(key, 0.0) + probability / 3.0
        total = sum(updated.values())
        if total <= 0:
            raise ValueError('Arrivals incompatible with finite-lead receipt belief')
        self.states = {state: p / total for state, p in sorted(updated.items())}
        if any((abs(sum(state) - transit) > 1e-07 for state in self.states)):
            raise ValueError('Posterior transit does not match public total')
        self.previous_transit = float(transit)

    def sample(self, paths, rng, stratified):
        probability = np.array(list(self.states.values()))
        cumulative = np.cumsum(probability)
        cumulative[-1] = 1.0
        if stratified:
            uniforms = (np.arange(paths) + rng.random(paths)) / paths
            rng.shuffle(uniforms)
        else:
            uniforms = rng.random(paths)
        indices = np.searchsorted(cumulative, uniforms)
        return np.array(list(self.states))[indices]

class BeliefProjection(ParentPolicy):
    belief_setting = dict(baseline=True)

    def reset(self):
        super().reset()
        self.receipt_belief = ReceiptBelief()
        self.receipt_rng = np.random.default_rng(1050429)

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time != 2 or self.belief_setting.get('baseline', False):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            self.receipt_belief.observe(previous_order, previous_arrivals, in_transit_total)
            if self.prediction is not None:
                self._learn(previous_demand, *self.prediction)
            self.history.append(previous_demand)
            if previous_order > 0:
                lost = self.last_transit + previous_order - previous_arrivals - in_transit_total
                if lost > 0.5 * previous_order:
                    self.destroyed += 1.0
                else:
                    self.survived += 1.0
                    self.surviving_orders[period - 1] = previous_order
        self.last_transit = in_transit_total
        h = self.rollout['horizon']
        k = self.rollout['paths']
        f, s = self._supply_forecast(self.history, h + 2)
        self.prediction = (f[:, 0].copy(), s.copy())
        w = np.exp(self.log_weights)
        w /= w.sum()
        member = self.roll_rng.choice(len(w), k, p=w)
        noise = self.roll_rng.standard_t(5, size=(k, h)) * np.sqrt(3 / 5)
        demand = np.maximum(0.0, f[member, :h] + s[member, None] * noise)
        rate = np.maximum(1.0, np.convolve(w @ f, np.ones(3) / 3, mode='valid')[:h])
        survival = np.clip(self.survived / (self.survived + self.destroyed), 0.2, 1.0)
        alive = self.roll_rng.random((k, h)) < survival
        leads = self.roll_rng.integers(1, 4, size=(k, h))
        for age in (1, 2, 3):
            self.roll_rng.integers(0, 4 - age, size=k)
        committed = np.zeros((k, h + 3))
        committed[:, :3] = self.receipt_belief.sample(k, self.receipt_rng, self.belief_setting['stratified'])
        self.surviving_orders = {t: q for t, q in self.surviving_orders.items() if t >= period - 3}
        parameters = self.refined_parameters if self.supply_extension['refine'] else self.parameters
        theta = parameters[:, 0, None]
        cap = parameters[:, 1, None]
        n = len(theta)
        stock = np.full((n, k), on_hand_inventory, dtype=float)
        transit = np.full((n, k), in_transit_total, dtype=float)
        receipts = np.broadcast_to(committed, (n, k, h + 3)).copy()
        reward = np.zeros((n, k))
        actions = None
        for step in range(h):
            order = np.floor(np.maximum(0.0, np.minimum(cap * rate[step], theta * rate[step] - stock - transit)))
            if step == 0:
                actions = order[:, 0].copy()
            finite = order * alive[None, :, step]
            transit += finite
            for delay in (1, 2, 3):
                receipts[:, :, step + delay] += finite * (leads[None, :, step] == delay)
            received = receipts[:, :, step]
            transit -= received
            stock += received
            sold = np.minimum(stock, demand[None, :, step])
            stock -= sold
            reward += profit_per_unit * sold - holding_cost_per_unit * stock
        reward -= holding_cost_per_unit * stock * stock / (2 * rate[-1])
        values = reward.mean(axis=1)
        if self.rollout['blend']:
            scale = max(1e-09, float(np.median(reward.std(axis=1, ddof=1) / np.sqrt(k))))
            blend = np.exp(np.maximum(-700.0, (values - values.max()) / scale))
            blend /= blend.sum()
            return max(0, int(np.floor(blend @ actions)))
        return int(actions[np.argmax(values)])

class InventoryBenchPolicy(BeliefProjection):
    belief_setting={'stratified': True}

ParentPolicy=InventoryBenchPolicy

"""Common-random-number-aware regularization of capped-policy rollout values.

Only the stochastic action aggregation changes. The exact receipt filter,
forecast distribution, simulation paths, candidate controls, and horizon are
identical to the complete frozen receipt parent. Paired uncertainty removes
pathwise reward shocks shared by candidate policies before setting softmax
regularization. One global multiplier is selected on generated training.
"""
import numpy as np

def paired_action(reward, actions, temperature):
    values = reward.mean(axis=1)
    best = int(np.argmax(values))
    differences = reward - reward[best:best + 1]
    scale = max(1e-09, float(temperature * np.median(differences.std(axis=1, ddof=1) / np.sqrt(reward.shape[1]))))
    weights = np.exp(np.maximum(-700.0, (values - values[best]) / scale))
    weights /= weights.sum()
    return max(0, int(np.floor(weights @ actions)))

class PairedProjection(ParentPolicy):
    regularization_setting = dict(baseline=True)

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time != 2 or self.regularization_setting.get('baseline', False):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            self.receipt_belief.observe(previous_order, previous_arrivals, in_transit_total)
            if self.prediction is not None:
                self._learn(previous_demand, *self.prediction)
            self.history.append(previous_demand)
            if previous_order > 0:
                lost = self.last_transit + previous_order - previous_arrivals - in_transit_total
                if lost > 0.5 * previous_order:
                    self.destroyed += 1.0
                else:
                    self.survived += 1.0
                    self.surviving_orders[period - 1] = previous_order
        self.last_transit = in_transit_total
        h = self.rollout['horizon']
        k = self.rollout['paths']
        f, s = self._supply_forecast(self.history, h + 2)
        self.prediction = (f[:, 0].copy(), s.copy())
        w = np.exp(self.log_weights)
        w /= w.sum()
        member = self.roll_rng.choice(len(w), k, p=w)
        noise = self.roll_rng.standard_t(5, size=(k, h)) * np.sqrt(3 / 5)
        demand = np.maximum(0.0, f[member, :h] + s[member, None] * noise)
        rate = np.maximum(1.0, np.convolve(w @ f, np.ones(3) / 3, mode='valid')[:h])
        survival = np.clip(self.survived / (self.survived + self.destroyed), 0.2, 1.0)
        alive = self.roll_rng.random((k, h)) < survival
        leads = self.roll_rng.integers(1, 4, size=(k, h))
        for age in (1, 2, 3):
            self.roll_rng.integers(0, 4 - age, size=k)
        committed = np.zeros((k, h + 3))
        committed[:, :3] = self.receipt_belief.sample(k, self.receipt_rng, self.belief_setting['stratified'])
        self.surviving_orders = {t: q for t, q in self.surviving_orders.items() if t >= period - 3}
        parameters = self.refined_parameters if self.supply_extension['refine'] else self.parameters
        theta = parameters[:, 0, None]
        cap = parameters[:, 1, None]
        n = len(theta)
        stock = np.full((n, k), on_hand_inventory, dtype=float)
        transit = np.full((n, k), in_transit_total, dtype=float)
        receipts = np.broadcast_to(committed, (n, k, h + 3)).copy()
        reward = np.zeros((n, k))
        actions = None
        for step in range(h):
            order = np.floor(np.maximum(0.0, np.minimum(cap * rate[step], theta * rate[step] - stock - transit)))
            if step == 0:
                actions = order[:, 0].copy()
            finite = order * alive[None, :, step]
            transit += finite
            for delay in (1, 2, 3):
                receipts[:, :, step + delay] += finite * (leads[None, :, step] == delay)
            received = receipts[:, :, step]
            transit -= received
            stock += received
            sold = np.minimum(stock, demand[None, :, step])
            stock -= sold
            reward += profit_per_unit * sold - holding_cost_per_unit * stock
        reward -= holding_cost_per_unit * stock * stock / (2 * rate[-1])
        values = reward.mean(axis=1)
        return paired_action(reward, actions, self.regularization_setting['temperature'])

class InventoryBenchPolicy(PairedProjection):
    regularization_setting={'temperature': 2.0}

ParentPolicy=InventoryBenchPolicy

"""Use the documented supply law or stronger shrinkage in stochastic rollout.
Uniform lead times1,2,3,infinity imply known survival0.75. The exact paired
parent instead updates a Beta(3,1) mean from past positive-order receipts.
Only this survival estimate changes; forecasts, receipt inference, horizon,
paths, action aggregation, and all deterministic-lead branches are inherited.
"""
import numpy as np

def paired_action(reward, actions, temperature):
    values = reward.mean(axis=1)
    best = int(np.argmax(values))
    differences = reward - reward[best:best + 1]
    scale = max(1e-09, float(temperature * np.median(differences.std(axis=1, ddof=1) / np.sqrt(reward.shape[1]))))
    weights = np.exp(np.maximum(-700.0, (values - values[best]) / scale))
    weights /= weights.sum()
    return max(0, int(np.floor(weights @ actions)))

class SupplyPriorProjection(ParentPolicy):
    supply_prior_setting = dict(baseline=True)

    def reset(self):
        super().reset()
        if self.promised_lead_time == 2 and (not self.supply_prior_setting.get('baseline', False)):
            total = self.supply_prior_setting.get('prior_count', 4.0)
            self.survived = 0.75 * total
            self.destroyed = 0.25 * total

    def get_order(self, period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit):
        if self.promised_lead_time != 2 or self.supply_prior_setting.get('baseline', False):
            return super().get_order(period, current_date, on_hand_inventory, in_transit_total, previous_demand, previous_order, previous_arrivals, profit_per_unit, holding_cost_per_unit)
        if period > 1:
            self.receipt_belief.observe(previous_order, previous_arrivals, in_transit_total)
            if self.prediction is not None:
                self._learn(previous_demand, *self.prediction)
            self.history.append(previous_demand)
            if previous_order > 0:
                lost = self.last_transit + previous_order - previous_arrivals - in_transit_total
                if lost > 0.5 * previous_order:
                    self.destroyed += 1.0
                else:
                    self.survived += 1.0
                    self.surviving_orders[period - 1] = previous_order
        self.last_transit = in_transit_total
        h = self.rollout['horizon']
        k = self.rollout['paths']
        f, s = self._supply_forecast(self.history, h + 2)
        self.prediction = (f[:, 0].copy(), s.copy())
        w = np.exp(self.log_weights)
        w /= w.sum()
        member = self.roll_rng.choice(len(w), k, p=w)
        noise = self.roll_rng.standard_t(5, size=(k, h)) * np.sqrt(3 / 5)
        demand = np.maximum(0.0, f[member, :h] + s[member, None] * noise)
        rate = np.maximum(1.0, np.convolve(w @ f, np.ones(3) / 3, mode='valid')[:h])
        survival = 0.75 if self.supply_prior_setting.get('known_probability', False) else np.clip(self.survived / (self.survived + self.destroyed), 0.2, 1.0)
        alive = self.roll_rng.random((k, h)) < survival
        leads = self.roll_rng.integers(1, 4, size=(k, h))
        for age in (1, 2, 3):
            self.roll_rng.integers(0, 4 - age, size=k)
        committed = np.zeros((k, h + 3))
        committed[:, :3] = self.receipt_belief.sample(k, self.receipt_rng, self.belief_setting['stratified'])
        self.surviving_orders = {t: q for t, q in self.surviving_orders.items() if t >= period - 3}
        parameters = self.refined_parameters if self.supply_extension['refine'] else self.parameters
        theta = parameters[:, 0, None]
        cap = parameters[:, 1, None]
        n = len(theta)
        stock = np.full((n, k), on_hand_inventory, dtype=float)
        transit = np.full((n, k), in_transit_total, dtype=float)
        receipts = np.broadcast_to(committed, (n, k, h + 3)).copy()
        reward = np.zeros((n, k))
        actions = None
        for step in range(h):
            order = np.floor(np.maximum(0.0, np.minimum(cap * rate[step], theta * rate[step] - stock - transit)))
            if step == 0:
                actions = order[:, 0].copy()
            finite = order * alive[None, :, step]
            transit += finite
            for delay in (1, 2, 3):
                receipts[:, :, step + delay] += finite * (leads[None, :, step] == delay)
            received = receipts[:, :, step]
            transit -= received
            stock += received
            sold = np.minimum(stock, demand[None, :, step])
            stock -= sold
            reward += profit_per_unit * sold - holding_cost_per_unit * stock
        reward -= holding_cost_per_unit * stock * stock / (2 * rate[-1])
        values = reward.mean(axis=1)
        return paired_action(reward, actions, 2.0)

class InventoryBenchPolicy(SupplyPriorProjection):
    supply_prior_setting={'known_probability': True}

ParentPolicy=InventoryBenchPolicy
"""Execution-only fused rollout; frozen policy sources remain unchanged.

No fastmath, forecasts, RNG changes or parameter changes. The original policy
is the submission artifact and the full official cold-replay reference.
"""
import ast, inspect, textwrap
import numpy as np
from numba import njit

@njit(cache=False)
def rollout_kernel(on_hand, transit_total, committed, parameters, factors, rate, alive, leads, demand, profit, holding):
    n = len(parameters)
    k, h = demand.shape
    rewards = np.zeros((n, k))
    actions = np.zeros(n)
    for j in range(n):
        theta = parameters[j, 0]
        cap = parameters[j, 1]
        for path in range(k):
            stock = float(on_hand)
            transit = float(transit_total)
            receipts = committed[path].copy()
            reward = 0.0
            for step in range(h):
                desired = max(0.0, min(cap * rate[step], theta * rate[step] - stock - transit))
                order = np.floor(desired * factors[j]) if step == 0 else np.floor(desired)
                if step == 0 and path == 0:
                    actions[j] = order
                finite = order * alive[path, step]
                transit += finite
                for delay in range(1, 4):
                    receipts[step + delay] += finite * (leads[path, step] == delay)
                received = receipts[step]
                transit -= received
                stock += received
                sold = min(stock, demand[path, step])
                stock -= sold
                reward += profit * sold - holding * stock
            reward -= holding * stock * stock / (2 * rate[-1])
            rewards[j, path] = reward
    return (rewards, actions)

def make_method(original, is_candidate):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    node = tree.body[0]
    assert isinstance(node.body[0], ast.If)
    node.body.pop(0)
    start = next((i for i, n in enumerate(node.body) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'stock' for t in n.targets))))
    stop = next((i for i, n in enumerate(node.body) if i > start and isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'values' for t in n.targets))))
    factors = 'first_factor[:,0]' if is_candidate else 'np.ones(n)'
    replacement = ast.parse('reward,actions=rollout_kernel(on_hand_inventory,in_transit_total,committed,parameters,' + factors + ',rate,alive,leads,demand,profit_per_unit,holding_cost_per_unit)').body
    node.body[start:stop] = replacement
    namespace = dict(original.__globals__, rollout_kernel=rollout_kernel)
    exec(compile(ast.fix_missing_locations(tree), '<fused_rollout>', 'exec'), namespace)
    method = namespace['get_order']

    def dispatch(self, *args, **kwargs):
        if self.promised_lead_time != 2:
            return original(self, *args, **kwargs)
        if is_candidate and self.first_action_setting.get('baseline', False):
            return FastParent.get_order(self, *args, **kwargs)
        return method(self, *args, **kwargs)
    return dispatch

class FastParent(ParentPolicy):
    get_order = make_method(SupplyPriorProjection.get_order, False)

InventoryBenchPolicy=FastParent

ParentPolicy=InventoryBenchPolicy

"""Training-only stochastic rollout paths; all other coefficients inherited."""

class SamplingPolicy(ParentPolicy):
    sampling_setting = dict(paths=32)

    def __init__(self, *args, **kwargs):
        self.rollout = dict(ParentPolicy.rollout)
        self.rollout['paths'] = self.sampling_setting['paths']
        super().__init__(*args, **kwargs)

class InventoryBenchPolicy(SamplingPolicy):
    sampling_setting={'paths': 128}
