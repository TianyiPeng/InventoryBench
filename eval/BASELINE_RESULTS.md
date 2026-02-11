# YesterdayDemandPolicy Baseline Results

This document summarizes the performance of the **YesterdayDemandPolicy** baseline across all 1,320 InventoryBench instances.

## Policy Description

**YesterdayDemandPolicy**: A simple heuristic that orders `max(0, yesterday's_demand - current_inventory)`. This policy:
- Uses yesterday's demand as a forecast for today
- Adjusts for current on-hand inventory
- Ignores lead times and in-transit orders

This serves as a naive baseline that any sophisticated policy should outperform.

## Complete Results

### Real Trajectory (H&M Retail Data) - 600 instances

| Lead Time Setting | Normalized Score | Service Level | Notes |
|-------------------|-----------------|---------------|-------|
| **lead_time_0** (immediate) | **0.7555** | 84.05% | Best performance - no forecasting needed |
| **lead_time_4** (fixed delay) | **0.1381** | 62.78% | Poor - needs lead time planning |
| **lead_time_stochastic** (2±random) | **0.3156** | 49.17% | Challenging - random delays & losses |

### Synthetic Trajectory - 720 instances

| Lead Time Setting | Normalized Score | Service Level | Notes |
|-------------------|-----------------|---------------|-------|
| **lead_time_0** (immediate) | **0.7993** | 86.89% | Best performance - predictable patterns |
| **lead_time_4** (fixed delay) | **0.5273** | 69.80% | Moderate - synthetic more regular |
| **lead_time_stochastic** (2±random) | **0.3798** | 51.52% | Still challenging despite regularity |

## Key Insights

1. **Zero Lead Time Performance (75-80%)**
   - When orders arrive immediately, yesterday's demand is a good predictor
   - Achieves 75-80% of perfect foresight performance
   - Service levels around 84-87%

2. **Impact of Lead Times**
   - **Real data with lead_time_4**: Drops to **13.8%** (catastrophic)
   - **Synthetic with lead_time_4**: Maintains **52.7%** (better regularity)
   - Lead time planning is critical for real-world performance

3. **Stochastic Lead Times**
   - **Real data**: **31.6%** performance, **49%** service level
   - **Synthetic data**: **38.0%** performance, **52%** service level
   - Random delays and lost orders significantly impact naive policies
   - Service level drops below 50% - barely satisfying half of demand

4. **Real vs Synthetic**
   - Real H&M data is substantially harder than synthetic patterns
   - With lead_time_4: Real (13.8%) vs Synthetic (52.7%) - **4x difference**
   - Real retail data has more volatile, unpredictable demand

5. **Negative Returns**
   - Some instances achieve negative normalized scores (e.g., -4.51 min on real/lead_time_4)
   - Holding costs can exceed profits when over-ordering with long lead times
   - YesterdayDemandPolicy doesn't account for in-transit inventory

## Benchmark Difficulty

The wide performance variance (13.8% to 79.9%) demonstrates that:
- InventoryBench spans easy to extremely challenging scenarios
- Simple heuristics fail on realistic supply chain conditions
- Sophisticated policies must handle lead time uncertainty and demand forecasting

## Recommendations for Competitive Policies

To significantly outperform this baseline:

1. **Lead Time Awareness**: Track in-transit orders, don't re-order what's already coming
2. **Demand Forecasting**: Use moving averages, exponential smoothing, or ML models instead of yesterday's demand
3. **Safety Stock**: Maintain buffer inventory proportional to lead time and demand variance
4. **Adaptive Ordering**: Adjust order quantities based on observed lead time patterns
5. **Cost-Aware Decisions**: Balance holding costs vs stockout costs explicitly

## Files Generated

All results are saved in `eval/example_policy_test/` with the structure:
```
eval/example_policy_test/
├── real_trajectory/
│   ├── lead_time_0/
│   │   ├── {instance_id}/results.csv
│   │   └── evaluation_report.json
│   ├── lead_time_4/
│   │   └── ...
│   └── lead_time_stochastic/
│       └── ...
└── synthetic_trajectory/
    ├── lead_time_0/
    ├── lead_time_4/
    └── lead_time_stochastic/
```

Each `evaluation_report.json` contains:
- Per-instance detailed metrics
- Aggregate statistics (mean, median, std, range)
- Final normalized score
- Service level statistics
