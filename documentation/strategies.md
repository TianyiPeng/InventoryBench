# The Four Strategies

This document describes the four inventory management strategies implemented in
the OR Agent framework. Each strategy represents a different division of labor
between classical Operations Research (OR) algorithms and Large Language Models
(LLMs).

---

## 1. OR (Base-Stock Only)

**Type:** Pure statistical baseline -- no LLM involvement.

### Algorithm: Capped Base-Stock Policy

The OR strategy estimates demand statistics from all available historical samples
and computes a base-stock level each period.

**Demand estimation** from historical samples (initial training data plus all
observed demands so far):

```
empirical_mean = (1/n) * sum(samples)
empirical_std  = sqrt( (1/(n-1)) * sum((sample_i - empirical_mean)^2) )
```

**Lead-time-adjusted parameters:**

```
mu_hat    = (1 + L) * empirical_mean
sigma_hat = sqrt(1 + L) * empirical_std
```

where `L` is the promised lead time in periods.

**Safety factor and base-stock level:**

```
rho  = p / (p + h)                 # critical fractile
z*   = inverse_normal_CDF(rho)     # safety factor
B_t  = mu_hat + z* * sigma_hat     # base-stock level
```

**Order quantity (capped policy):**

```
order_uncapped = max(B_t - IP_t, 0)
cap            = mu_hat / (1 + L) + inverse_normal_CDF(0.95) * sigma_hat / sqrt(1 + L)
q_t            = min(order_uncapped, cap)
```

where `IP_t` is the inventory position (on-hand plus in-transit).

### Limitations

- Assumes demand is stationary and independently identically distributed (IID).
- Uses the promised lead time, not the actual observed lead time.
- Treats all historical samples equally with no recency weighting.
- Cannot detect lost orders, regime shifts, seasonality, or supply disruptions.

### Demo Script

```
python scripts/run_or.py \
  --demand-file <path/to/test.csv> \
  --real-instance-train <path/to/train.csv> \
  --promised-lead-time 0 \
  --policy capped
```

---

## 2. LLM-Only

**Type:** The LLM receives the full game state each period and directly decides
order quantities with no OR algorithm involved.

### How It Works

1. A detailed system prompt is constructed covering:
   - **Role and objective:** Maximize total reward (profit from sales minus
     holding costs).
   - **Game mechanics:** The four-phase period execution sequence (VM decision,
     arrival resolution, demand resolution, period conclusion).
   - **Lead time definition:** Precise timing of when orders arrive and when
     arrivals become visible in observations.
   - **Inventory tracking:** How to interpret on-hand and in-transit quantities.
   - **Demand reasoning guidance:** Instructions to use world knowledge (product
     descriptions, calendar dates) to forecast demand.

2. Each period, the LLM receives the current observation (inventory levels,
   order history, demand history, product descriptions, calendar dates) and
   produces a JSON response:

```json
{
  "rationale": "Step-by-step reasoning...",
  "carry_over_insight": "New pattern detected or empty string",
  "action": {"item_id": quantity}
}
```

3. The **carry-over insight** mechanism provides cross-period memory. Insights
   from previous periods are prepended to future observations so the LLM can
   recall detected patterns (demand regime shifts, lead time discoveries, etc.).

### Strengths

- Can reason about contextual information (product categories, seasonal events,
  calendar dates).
- Can detect non-stationary patterns, regime shifts, and anomalies.
- Can handle unexpected events using world knowledge.

### Weaknesses

- Poorly calibrated to the cost structure; does not inherently compute the
  critical ratio or safety factors.
- Can be inconsistent across periods.
- Higher computational cost per decision (API call per period).

### Demo Script

```
python scripts/run_llm.py \
  --demand-file <path/to/test.csv> \
  --real-instance-train <path/to/train.csv> \
  --promised-lead-time 0 \
  --model <model_name>
```

---

## 3. OR-to-LLM (OR Recommends, LLM Decides)

**Type:** Hybrid strategy where the OR algorithm provides a recommendation each
period, and the LLM makes the final decision with the option to follow, adjust,
or override.

### How It Works

1. The system prompt includes everything from the LLM-only strategy **plus** a
   complete explanation of the OR algorithm's mathematical details and its
   limitations.

2. Each period, the observation is augmented with the OR algorithm's
   recommendation, including:
   - Recommended order quantity
   - Base-stock level (`B_t`)
   - Current inventory position (`IP_t`)
   - Demand statistics (`mu_hat`, `sigma_hat`, empirical mean/std)
   - Cap value
   - Critical fractile and safety factor

3. The LLM follows a **4-step decision checklist**:
   1. Use world knowledge and the product description to form a demand outlook.
   2. Reconcile on-hand inventory plus pipeline; flag overdue or lost shipments.
   3. Inspect the OR recommendation and decide how to adapt it.
   4. Justify the final order quantity.

4. The carry-over insight mechanism operates identically to the LLM-only
   strategy, providing cross-period memory.

### Collaboration Strategy

The LLM treats the OR recommendation as a data-driven starting point and
considers overriding it when:

- The actual lead time differs from the promised lead time (inferred from
  concluded period messages).
- A demand regime change is detected from recent history.
- Seasonal or calendar-based demand effects apply (world knowledge).
- Lost shipments or pipeline anomalies are identified.

### Why This Strategy Excels

- Combines the statistical rigor of the OR policy with the contextual reasoning
  of an LLM.
- The LLM can detect and correct for situations where the OR algorithm's
  assumptions break down (non-stationarity, lost orders, seasonality).
- Best overall performance across the benchmark, particularly in stochastic lead
  time settings where lost orders must be detected.

### Demo Script

```
python scripts/run_or_to_llm.py \
  --demand-file <path/to/test.csv> \
  --real-instance-train <path/to/train.csv> \
  --promised-lead-time 0 \
  --model <model_name>
```

---

## 4. LLM-to-OR (LLM Proposes Parameters, OR Executes)

**Type:** The LLM analyzes the game state and proposes parameters for the OR
algorithm, which then computes the order quantity.

### How It Works

1. The LLM receives the full observation (same information as the other
   strategies) and outputs a structured JSON specifying **parameters** rather
   than a direct order quantity.

2. For each item, the LLM proposes:
   - **L** (lead time estimate)
   - **mu_hat** (mean demand estimate for the review-plus-lead-time period)
   - **sigma_hat** (standard deviation estimate)

3. For each parameter, the LLM selects a **method**:

   | Method     | Description                                               |
   |------------|-----------------------------------------------------------|
   | `default`  | Use all historical samples with empirical statistics      |
   | `explicit` | Provide a specific numeric value directly                 |
   | `recent_N` | Use only the last N periods of observed demand            |

   For lead time specifically, there is also a `calculate` method that infers
   the actual lead time from concluded period messages.

4. The OR backend then uses these parameters to compute:

```
base_stock = mu_hat + z* * sigma_hat
order      = max(0, min(base_stock - IP_t, cap))
```

   using the same capped base-stock formula as the pure OR strategy.

### Example LLM Output

```json
{
  "rationale": "Demand has shifted upward; using recent 5 periods...",
  "carry_over_insight": "",
  "parameters": {
    "item_id": {
      "L": {"method": "explicit", "value": 4},
      "mu_hat": {"method": "recent_N", "N": 5},
      "sigma_hat": {"method": "recent_N", "N": 5}
    }
  }
}
```

### Advantages

- Maintains the structural guarantees and cost-calibration of the OR algorithm
  (critical ratio, safety factor, capping).
- Leverages the LLM's pattern recognition to improve parameter estimation
  (detecting regime shifts, adjusting lead time beliefs).
- Output is more constrained than direct order quantities, reducing the chance
  of erratic LLM behavior.

### Best Use Case

This strategy tends to dominate in **deterministic lead time** settings, where
the OR algorithm's structural framework is most reliable and the LLM's main
contribution is improving demand parameter estimation.

### Demo Script

```
python scripts/run_llm_to_or.py \
  --demand-file <path/to/test.csv> \
  --real-instance-train <path/to/train.csv> \
  --promised-lead-time 0 \
  --model <model_name>
```

---

## Performance Summary

| Setting                  | Best Strategy  | Explanation                                         |
|--------------------------|----------------|-----------------------------------------------------|
| Deterministic lead times | LLM-to-OR      | OR structure is reliable; LLM improves parameters   |
| Stochastic lead times    | OR-to-LLM      | LLM detects lost orders and pipeline anomalies      |
| Stationary demand        | OR (baseline)  | Strong baseline when IID assumptions hold           |
| Non-stationary demand    | OR-to-LLM      | LLM adapts to regime shifts and seasonality         |
| Cost calibration         | OR / LLM-to-OR | Critical ratio math handled by OR algorithm         |

**Key takeaways:**

- The pure **OR** strategy is a strong baseline for stationary patterns and
  deterministic lead times, but breaks down under non-stationarity and supply
  disruptions.
- **LLM-only** struggles with cost calibration because it does not inherently
  compute the critical ratio or safety factors.
- **OR-to-LLM** provides the best overall performance by combining OR's
  statistical grounding with the LLM's ability to reason about context, detect
  anomalies, and override when assumptions are violated.
- **LLM-to-OR** is strongest under deterministic lead times, where the OR
  framework's structural properties are preserved and the LLM focuses on
  improving demand parameter estimates.
