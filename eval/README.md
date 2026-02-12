# InventoryBench Evaluation Framework

This directory contains tools for evaluating inventory policies on InventoryBench.

**Website & Leaderboard:** [https://tianyipeng.github.io/InventoryBench/](https://tianyipeng.github.io/InventoryBench/)

## Overview

Every submission follows this structure:
```
my_policy/
├── results/              # Instance-level order decisions
│   ├── real_trajectory/
│   │   ├── lead_time_0/{instance_id}/results.csv
│   │   ├── lead_time_4/{instance_id}/results.csv
│   │   └── lead_time_stochastic/{instance_id}/results.csv
│   └── synthetic_trajectory/
│       └── ...
├── scores.json           # Evaluation summary (auto-generated)
└── README.md             # Method description
```

## Files

| File | Description |
|------|-------------|
| `policy_template.py` | Base class and example policy implementation |
| `run_baseline_policy.py` | Runs a policy on all 1,320 benchmark instances |
| `evaluate_results.py` | Computes normalized scores from results |

## Quick Start

### Step 1: Run Your Policy

```bash
python eval/run_baseline_policy.py --output-dir my_policy
```

This generates `my_policy/results/` with a `results.csv` per instance (columns: `period, order_quantity`).

### Step 2: Evaluate

```bash
python eval/evaluate_results.py --submission-dir my_policy
```

This reads `my_policy/results/`, computes normalized scores, and saves `my_policy/scores.json`.

### Step 3: Implement Your Own Policy

Create a class inheriting from `InventoryPolicy`:

```python
from policy_template import InventoryPolicy

class MyPolicy(InventoryPolicy):
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
        # Your logic here
        return order_quantity
```

Then update the import in `run_baseline_policy.py`:
```python
from my_policy import MyPolicy
# Replace ExamplePolicy(...) with MyPolicy(...)
```

## Policy Interface

### Initialization Context

Your policy receives at initialization:

| Parameter | Description |
|-----------|-------------|
| `item_id` | SKU identifier |
| `initial_samples` | List of `(date, demand)` tuples from `train.csv` |
| `promised_lead_time` | 0, 2, or 4 periods (auto-detected from folder) |
| `profit_per_unit` | Profit per unit sold |
| `holding_cost_per_unit` | Holding cost per unit per period |
| `product_description` | Product description (real trajectories only) |

### Per-Period Inputs

| Parameter | Description |
|-----------|-------------|
| `period` | Current period number (1-indexed) |
| `current_date` | Date string (e.g., "2019-07-01") |
| `on_hand_inventory` | Current inventory level |
| `in_transit_total` | Total units ordered but not yet arrived |
| `previous_demand` | Actual demand from last period |
| `previous_order` | Your order from last period |
| `previous_arrivals` | Units that arrived last period |
| `profit_per_unit` | Profit per unit sold |
| `holding_cost_per_unit` | Cost per unit held per period |

**Important:** You do NOT observe actual lead times. You must infer them from observed arrivals.

### Output

Return a non-negative number (converted to `int(max(0, value))`).

## Game Mechanics

Each period follows this execution sequence:

1. **Decision**: Your policy places an order
2. **Arrivals**: Orders scheduled for this period are added to inventory
3. **Demand**: Demand is satisfied from on-hand inventory
4. **Reward**: `profit * units_sold - holding_cost * ending_inventory`

Key points:
- Orders placed in period N arrive in period N+L (L = actual lead time)
- Actual lead times may differ from promised (especially in `lead_time_stochastic/`)
- Some orders may be lost (lead_time = infinity) and never arrive
- Real trajectories have 47 periods; synthetic have 50

## Evaluation Metric

**Normalized Reward** (primary metric):
```
normalized_reward = max(0, total_reward / perfect_foresight_reward)
```

- `total_reward` = cumulative (profit * sold - holding_cost * inventory)
- `perfect_foresight_reward` = profit * total_demand (clairvoyant upper bound)

**Leaderboard score** = mean normalized reward across all 1,320 instances.

## Submission

### 1. Run your policy
```bash
python eval/run_baseline_policy.py --output-dir my_policy
```

### 2. Evaluate
```bash
python eval/evaluate_results.py --submission-dir my_policy
```
This generates `my_policy/scores.json` with overall score, per-batch scores (6 batches), and detailed statistics.

### 3. Package your submission

Your final submission folder should contain:
```
my_policy/
├── results/                    # Full results structure
│   ├── real_trajectory/
│   │   ├── lead_time_0/
│   │   │   ├── 108775044/
│   │   │   │   └── results.csv
│   │   │   └── .../
│   │   ├── lead_time_4/
│   │   └── lead_time_stochastic/
│   └── synthetic_trajectory/
│       ├── lead_time_0/
│       ├── lead_time_4/
│       └── lead_time_stochastic/
├── scores.json                 # Evaluation summary
└── README.md                   # Method description
```

**README.md should include:**
- Brief description of your approach
- Key design decisions and algorithms
- Links to code repositories (if available)
- Links to related papers (if applicable)
- Any relevant implementation notes

### 4. Submit

Zip the whole folder and submit via GitHub PR or email to tianyipeng95@gmail.com

## FAQ

**Can I use the promised lead time?**
Yes. It's provided at initialization. But actual lead times may differ, so adapt based on observed arrivals.

**What if my policy needs more historical data?**
You only get 5 samples from `train.csv`. Design your policy to learn online.

**Can I use external data (e.g., holiday calendars)?**
Yes. `current_date` is provided. Use any world knowledge you want, but don't directly use future demand.

**What happens if I order too much?**
Excess inventory incurs holding costs every period.

**What if demand exceeds inventory?**
You sell what you have (stockout). Unmet demand has no direct penalty, but you lose potential profit.

## Troubleshooting

| Error | Fix |
|-------|-----|
| "No results/ subfolder found" | Ensure your submission directory contains a `results/` subfolder |
| "No matching batches found" | Check that `results/` has `real_trajectory/` and/or `synthetic_trajectory/` subdirectories |
| "Period mismatch" | Ensure `results.csv` has exactly the right number of rows with correct period numbers |
| ImportError | Run from repository root: `cd /path/to/InventoryBench` |
