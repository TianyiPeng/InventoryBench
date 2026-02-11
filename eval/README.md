# InventoryBench Evaluation Framework

This directory contains tools for evaluating inventory policies on InventoryBench.

## 📋 Overview

**Submission Process:**
1. Implement your policy following the `InventoryPolicy` interface
2. Run your policy to generate `results/` folder with order decisions
3. Run evaluation script to compute normalized average score
4. Submit `results/` folder + evaluation report

## 🏗️ Files

- **`policy_template.py`** - Base class and example policy implementation
- **`run_baseline_policy.py`** - Runs example baseline on all benchmark instances
- **`evaluate_results.py`** - Computes normalized scores from results

## 🚀 Quick Start

### 1. Test the Baseline

Run the example "Yesterday's Demand" baseline:

```bash
# From repository root
python eval/run_baseline_policy.py \
    --benchmark-dir benchmark \
    --output-dir results/yesterday_demand
```

This will:
- Enumerate all 1,320 instances in `benchmark/`
- Run the baseline policy on each
- Output `results/yesterday_demand/` with same folder structure
- Each instance gets `results.csv` with columns: `period, order_quantity`

### 2. Evaluate Results

Compute normalized scores:

```bash
python eval/evaluate_results.py \
    --benchmark-dir benchmark \
    --results-dir results/yesterday_demand
```

This will:
- Auto-detect all 6 batches (real/synthetic × lead_time_0/4/stochastic)
- Simulate game mechanics for each instance
- Compute rewards (profit - holding costs)
- Normalize by perfect foresight upper bound (clamped to [0, 1])
- Report per-batch scores and overall average

**Output:**
```
=== OVERALL EVALUATION SUMMARY ===

Total Instances: 1320

Batch Scores:
  real_trajectory/lead_time_0             : 0.7555 (200 instances)
  real_trajectory/lead_time_4             : 0.3260 (200 instances)
  real_trajectory/lead_time_stochastic    : 0.3386 (200 instances)
  synthetic_trajectory/lead_time_0        : 0.7993 (240 instances)
  synthetic_trajectory/lead_time_4        : 0.5377 (240 instances)
  synthetic_trajectory/lead_time_stochastic: 0.3819 (240 instances)

>>> OVERALL SCORE: 0.5277 <<<
```

A detailed JSON file (`evaluation_summary.json`) is also created with full statistics for each batch.

### 3. Implement Your Own Policy

Create a new file (e.g., `my_policy.py`):

```python
from policy_template import InventoryPolicy

class MyPolicy(InventoryPolicy):
    """Your custom inventory policy."""
    
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
        """Determine order quantity for current period."""
        
        # Your logic here
        # ...
        
        return order_quantity
```

Then modify `run_baseline_policy.py` to use your policy:
```python
from my_policy import MyPolicy

# In run_all_instances():
policy = MyPolicy(
    item_id=item_id,
    initial_samples=initial_samples,
    promised_lead_time=promised_lead_time,
    profit_per_unit=profit_per_unit,
    holding_cost_per_unit=holding_cost_per_unit
)
```

## 📊 Input Specifications

### Initial Context (per instance)

Your policy receives at initialization:

1. **Historical Samples** (from `train.csv`):
   ```python
   [
       ("2019-01-01", 108.0),  # (date, demand)
       ("2019-01-08", 124.0),
       ...
   ]
   ```

2. **Promised Lead Time**: Integer (0, 2, or 4 periods)
   - Auto-detected from folder structure:
     - `lead_time_0/` → 0 periods
     - `lead_time_4/` → 4 periods  
     - `lead_time_stochastic/` → 2 periods (promised, but actual varies)

3. **Cost Parameters**: `profit_per_unit`, `holding_cost_per_unit`

### Per-Period Inputs

At each period, your policy receives:

```python
period: int              # Current period number (1-indexed)
current_date: str        # Date for this period (e.g., "2019-07-01")
on_hand_inventory: float # Current inventory level
in_transit_total: float  # Total units ordered but not yet arrived
previous_demand: float   # Actual demand from last period
previous_order: float    # Your order from last period
previous_arrivals: float # Units that arrived last period
profit_per_unit: float   # Profit per unit sold
holding_cost_per_unit: float # Cost per unit held per period
```

**Important:** You do NOT observe actual lead times! You must infer them from observed arrivals.

### Output Format

Return a non-negative integer:
```python
order_quantity: float  # Will be converted to int(max(0, order_quantity))
```

## 🎮 Game Mechanics

Each period follows this execution sequence:

1. **Decision Phase**: Your policy places order
2. **Arrival Resolution**: Orders scheduled to arrive this period are added to inventory
3. **Demand Resolution**: Demand is satisfied from on-hand inventory
4. **Cost Accounting**: Reward = Profit × units_sold - HoldingCost × ending_inventory

**Key Points:**
- Orders placed in period N arrive in period N+L (where L = actual lead time)
- You observe arrivals with 1-period delay
- Actual lead times may differ from promised (esp. in `lead_time_stochastic/`)
- Some orders may be lost (lead_time = ∞) and never arrive

## 📈 Evaluation Metrics

**Primary Metric: Normalized Reward**
```
normalized_reward = total_reward / perfect_foresight_reward
```

Where:
- `total_reward = sum(profit × units_sold - holding_cost × ending_inventory)`
- `perfect_foresight_reward = profit × sum(all_demands)` (upper bound)

**Leaderboard Score:** Mean normalized reward across all 1,320 instances

**Other Metrics:**
- Service level: Fraction of demand satisfied
- Total reward: Absolute reward achieved
- Ending inventory: Final inventory level

## 📁 File Structure

**Input (Benchmark):**
```
benchmark/
├── real_trajectory/
│   ├── lead_time_0/
│   │   ├── 108775044/
│   │   │   ├── train.csv      # Historical samples
│   │   │   └── test.csv       # Ground truth trajectory
│   │   └── .../
│   ├── lead_time_4/
│   └── lead_time_stochastic/
└── synthetic_trajectory/
    └── .../
```

**Output (Results):**
```
results/my_policy/
├── real_trajectory/
│   ├── lead_time_0/
│   │   ├── 108775044/
│   │   │   └── results.csv    # Your order decisions
│   │   └── .../
│   ├── lead_time_4/
│   └── lead_time_stochastic/
├── synthetic_trajectory/
    └── .../
└── evaluation_report.json     # Detailed metrics
```

## 📝 CSV Formats

### train.csv (Input)
```csv
exact_dates_{item_id},demand_{item_id}
2019-01-01,108
2019-01-08,124
...
```

### test.csv (Input - Ground Truth)
```csv
exact_dates_{item_id},demand_{item_id},lead_time_{item_id},profit_{item_id},holding_cost_{item_id},description_{item_id}
2019-07-01,142,0,2.0,1.0,"Retail Product 108775044"
2019-07-08,156,0,2.0,1.0,"Retail Product 108775044"
...
```

**Note:** Your policy does NOT see the `lead_time` or `demand` columns during execution!

### results.csv (Your Output)
```csv
period,order_quantity
1,150
2,160
...
```

## 🎯 Submission Guidelines

**For Leaderboard Submission:**

1. **Run your policy** to generate `results/` folder
2. **Run evaluation**:
   ```bash
   python eval/evaluate_results.py \
       --benchmark-dir benchmark \
       --results-dir results/my_policy \
       --output-json results/my_policy_evaluation.json
   ```
   This generates a single JSON with:
   - Overall score across all 1,320 instances
   - Per-batch scores (6 batches)
   - Detailed statistics

3. **Submit:**
   - `results/` folder (zipped)
   - `my_policy_evaluation.json`
   - Short description of your method (PDF/markdown)
   - (Optional) Code for reproducibility

4. **Contact:** Submit via [GitHub PR/Issues] or email to [your_email]

## ❓ FAQ

**Q: Can I use the promised lead time in my policy?**  
A: Yes! The promised lead time is provided at initialization. However, actual lead times may differ (especially in stochastic settings), so your policy should adapt based on observed arrivals.

**Q: What if my policy needs more historical data?**  
A: You only get the 5 samples from `train.csv`. Design your policy to bootstrap from limited data and learn online.

**Q: Can I use external data (e.g., holiday calendars)?**  
A: Yes! The `current_date` string is provided. You can use any world knowledge or calendar information you want.

**Q: What happens if I order too much?**  
A: Excess inventory incurs holding costs every period, reducing your reward.

**Q: What if demand exceeds inventory?**  
A: You sell what you have (stockout). Unmet demand has no direct penalty, but you lose potential profit.

**Q: How are ties broken on the leaderboard?**  
A: By highest mean reward (absolute), then by service level.

## 🔧 Troubleshooting

**"No results.csv files found"**  
→ Check your output directory structure matches benchmark structure

**"Period mismatch"**  
→ Ensure your results.csv has exactly `num_periods` rows with correct period numbers

**"Missing required columns"**  
→ Check train.csv and test.csv have required columns for the item_id

**ImportError**  
→ Make sure you're running from repository root:
```bash
cd /path/to/OR_Agent
python eval/run_baseline_policy.py ...
```

## 📚 References

- **Benchmark Paper:** [Coming soon]
- **GitHub:** https://github.com/TianyiPeng/InventoryBench
- **Leaderboard:** [Website URL]

---

Good luck! 🚀
