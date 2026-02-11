# Benchmark Specification

## Overview

The OR Agent benchmark consists of **1,320 instances** designed to evaluate
inventory management strategies under a wide range of demand patterns, cost
structures, and lead time configurations.

| Dataset       | Instances | Source         | Purpose                                        |
|---------------|-----------|----------------|------------------------------------------------|
| **Synthetic** | 720       | Generated      | Controlled evaluation across diverse patterns  |
| **Real**      | 600       | H&M retail     | Real-world performance validation              |

---

## Instance Families

### Synthetic (720 Instances)

Synthetically generated demand trajectories covering 10 distinct demand
patterns, each with 4 parameter variants.

**Structure:**

- 10 demand patterns x 4 variants = **40 unique distributions**
- 2 independent demand realizations per variant
- 3 cost ratios per realization
- 3 lead time configurations
- **Total: 40 x 2 x 3 x 3 = 720 instances**

**Demand patterns:**

| ID   | Pattern                | Description                                        |
|------|------------------------|----------------------------------------------------|
| p01  | Stationary IID         | Constant distribution throughout all periods       |
| p02  | Mean Increase          | Sudden upward shift in mean at period 16           |
| p03  | Mean Decrease          | Sudden downward shift in mean at period 16         |
| p04  | Increasing Trend       | Gradual increase in demand over time               |
| p05  | Decreasing Trend       | Gradual decrease in demand over time               |
| p06  | Variance Change        | Change in demand variability at period 16          |
| p07  | Seasonal               | Periodic sinusoidal demand variation               |
| p08  | Multiple Changepoints  | Two changepoints creating three demand regimes     |
| p09  | Temporary Spike/Dip    | Temporary anomaly followed by return to baseline   |
| p10  | AR(1)                  | Autocorrelated demand process                      |

Each pattern has 4 variants (v1 through v4) with different parameterizations.
For example, Stationary IID includes Normal(100, 25), Normal(100, 40),
Normal(100, 15), and Uniform(50, 150).

**Fixed parameters:**

| Parameter                | Value    |
|--------------------------|----------|
| Test periods             | 50       |
| Training samples         | 5        |
| Changepoint location     | Period 16 (for single-changepoint patterns) |
| AR(1) initial value      | 100      |
| Random seed              | 42       |

**Training data generation:**

| Pattern Type                  | Train Generation Method                      |
|-------------------------------|----------------------------------------------|
| Stationary (p01)              | 5 IID samples from the distribution          |
| Changepoint (p02, p03, p06)   | 5 IID samples from the first segment only    |
| Multi-changepoint (p08)       | 5 IID samples from the first segment only    |
| Temporary spike/dip (p09)     | 5 IID samples from the first segment only    |
| Trend (p04, p05)              | Sequential samples at t = 1, 2, 3, 4, 5     |
| Seasonal (p07)                | Sequential samples at t = 1, 2, 3, 4, 5     |
| AR(1) (p10)                   | Sequential samples at t = 1, 2, 3, 4, 5     |

Realizations sharing the same index (r1 or r2) have identical training data and
demand trajectories. Only the cost ratio differs.

### Real (600 Instances)

Weekly sales data from H&M Group retail, covering 200 distinct product articles.

**Structure:**

- 200 H&M retail products (diverse categories: swimwear, knitwear, basics,
  accessories)
- Weekly sales data from 2019
- 5-week training period, 48-week test period
- 3 lead time configurations
- **Total: 200 x 3 = 600 instances**

**Data source characteristics:**

- Weekly aggregated transaction data
- Top 200 articles by total sales volume in 2019
- Product descriptions include category, garment type, and material information
- Calendar dates provided (enabling seasonal reasoning)

---

## Lead Time Configurations

Both synthetic and real datasets include three lead time settings:

| Setting         | Value                        | Description                                   |
|-----------------|------------------------------|-----------------------------------------------|
| `lead_time_0`   | L = 0                        | Immediate delivery; orders arrive same period  |
| `lead_time_4`   | L = 4                        | Fixed 4-period delay                           |
| `lead_time_stochastic` | L in {1, 2, 3, inf}  | Each order independently sampled; `inf` = lost order (never arrives) |

For the stochastic setting, each order independently has its lead time drawn
uniformly from {1, 2, 3, inf} with equal probability (25% each). Orders with
`inf` lead time represent lost shipments that never arrive.

The same random lead time sequence is used across all instances within each
dataset to ensure comparability.

---

## Cost Parameters

Three critical ratios control the relative penalty of understocking versus
overstocking:

| Critical Ratio (rho) | Profit (p) | Holding Cost (h) | Interpretation                  |
|-----------------------|------------|-------------------|---------------------------------|
| 0.50                  | 1          | 1                 | Balanced; stockout and holding equally costly |
| 0.80                  | 4          | 1                 | Understocking moderately costly  |
| 0.95                  | 19         | 1                 | Stockouts very costly (service-critical)      |

The critical ratio `rho = p / (p + h)` determines the optimal service level
in the newsvendor framework:

- **rho = 0.50:** Equal costs yield the median as the optimal order-up-to level.
- **rho = 0.80:** Higher penalty for lost sales pushes toward higher inventory.
- **rho = 0.95:** Near-stockout-free operation required; large safety stocks.

**Assignment:**

- **Synthetic:** Each variant has 2 realizations x 3 cost ratios = 6 instances.
  Named `r1_low`, `r1_med`, `r1_high`, `r2_low`, `r2_med`, `r2_high`.
- **Real:** Each of the 600 instances has a cost ratio independently assigned
  (seed = 42).

---

## Evaluation Metric

### Normalized Reward

The primary metric is the **normalized reward**, which compares the agent's
cumulative reward to an optimistic upper bound:

```
Normalized Reward = max(agent_reward / (p * sum(d_t)), 0)
```

**Components:**

- `agent_reward = sum_t (p * s_t - h * I_{t+1})` where `s_t = min(d_t, I_t + A_t)` is units sold and `I_{t+1}` is ending inventory.
- `p * sum(d_t)` is the **optimistic upper bound** assuming all demand is
  captured and there is zero ending inventory every period.
- The result is clipped at 0 to prevent extreme negative values.

This normalization provides a standardized performance measure across instances
with different demand scales and cost structures. A value of 1.0 represents
perfect performance (all demand met, no waste). Values below 1.0 reflect lost
sales, excess holding costs, or both.

---

## Data Format

Each benchmark instance is a directory containing two CSV files.

### train.csv

Contains 5 historical demand periods used as context for algorithms.

```csv
exact_dates_{item_id},demand_{item_id}
2019-01-07,162
2019-01-14,142
2019-01-21,115
2019-01-28,133
2019-02-04,118
```

- **Synthetic instances:** Dates are generic period identifiers (`Period_1`,
  `Period_2`, ...).
- **Real instances:** Dates are actual calendar dates in `YYYY-MM-DD` format.

### test.csv

Contains the evaluation periods with demand, configuration, and product
metadata.

```csv
exact_dates_{item_id},demand_{item_id},description_{item_id},lead_time_{item_id},profit_{item_id},holding_cost_{item_id}
2019/2/11,157,Strap top | Garment Upper body | ...,0,4,1
2019/2/18,244,Strap top | Garment Upper body | ...,0,4,1
...
```

**Columns:**

| Column                     | Description                                          |
|----------------------------|------------------------------------------------------|
| `exact_dates_{item_id}`    | Period identifier or actual date                     |
| `demand_{item_id}`         | Actual customer demand for the period                |
| `description_{item_id}`    | Product description (category, type, material)       |
| `lead_time_{item_id}`      | Order lead time (0, 4, or stochastic per-period values) |
| `profit_{item_id}`         | Profit earned per unit sold                          |
| `holding_cost_{item_id}`   | Cost per unit per period for holding inventory       |

The `{item_id}` suffix is the unique identifier for the product (e.g.,
`108775044` for an H&M article or a synthetic item ID).

---

## Directory Structure

```
benchmark/
  synthetic_trajectory/
    lead_time_0/
      p01_stationary_iid/
        v1_normal_100_25/
          r1_low/   (train.csv, test.csv)
          r1_med/   (train.csv, test.csv)
          r1_high/  (train.csv, test.csv)
          r2_low/   (train.csv, test.csv)
          r2_med/   (train.csv, test.csv)
          r2_high/  (train.csv, test.csv)
        v2_normal_100_40/
          ...
      p02_mean_increase/
        ...
    lead_time_4/
      ...
    lead_time_stochastic/
      ...
  real_trajectory/
    lead_time_0/
      108775044/  (train.csv, test.csv)
      111586001/  (train.csv, test.csv)
      ... (200 articles total)
    lead_time_4/
      ...
    lead_time_stochastic/
      ...
```

---

## Running the Benchmark

### Single Instance

Run all strategies on a single benchmark instance:

```bash
python scripts/benchmark_all_strategies.py \
  --directory <path/to/instance_directory> \
  --model <model_name>
```

The script:

1. Auto-detects the lead time setting from the directory path (`lead_time_0`,
   `lead_time_4`, or `lead_time_stochastic`).
2. Runs all five strategies: `or`, `llm`, `llm_to_or`, `or_to_llm`, and
   `perfect_score`.
3. Saves results to `benchmark_results.json` in the instance directory.

**Strategies:**

| Strategy       | Type           | Description                                |
|----------------|----------------|--------------------------------------------|
| `or`           | Deterministic  | Pure OR base-stock policy                  |
| `llm`          | LLM-based      | LLM directly decides order quantities      |
| `llm_to_or`    | LLM-based      | LLM proposes parameters, OR computes order |
| `or_to_llm`    | LLM-based      | OR recommends, LLM decides final order     |
| `perfect_score` | Deterministic | Theoretical upper bound (perfect foresight)|

### Batch Execution

Run the benchmark across multiple instances:

```bash
python scripts/run_batch_benchmark.py \
  --base-dir <path/to/benchmark_root> \
  --model <model_name>
```

This script discovers all instance directories under the base directory and
runs `benchmark_all_strategies.py` on each one. Instances that already have
complete results are skipped.

---

## Reproducibility

All synthetic data generation uses deterministic seeding:

- **Base random seed:** 42
- **Train seed per distribution:** `hash((42, pattern, variant, "train")) % 2^32`
- **Test seed per realization:** `hash((42, pattern, variant, realization_index)) % 2^32`

The stochastic lead time sequence is also generated from a fixed seed to ensure
identical supply disruption patterns across all instances within each dataset.
