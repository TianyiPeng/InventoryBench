# The Inventory Game

## Problem Formulation

The inventory game is a multi-period inventory control problem for a single
product (SKU). An agent must decide how many units to order each period to
maximize cumulative profit while managing holding costs.

### State

Each period `t`, the agent observes:

- **On-hand inventory** `I_t`: units currently available for sale.
- **In-transit inventory**: units ordered but not yet arrived.
- **Demand history**: all previously observed demand values.
- **Order history**: all previously placed orders and their arrival status.
- **Contextual information** `x_t`: product description, calendar dates, news
  events (when available).

### Action

The agent selects an order quantity `q_t >= 0` (non-negative integer).

### Lead Time

Orders placed in period `t` are scheduled to arrive in period `t + L`, where
`L` is the lead time. However:

- The actual lead time may differ from the promised lead time.
- In stochastic settings, each order independently receives a random lead time.
- Orders may be lost entirely (lead time = infinity), meaning they never arrive.

### Demand and Sales

- Demand `d_t` is realized and fully observed (uncensored) each period.
- Arrivals `A_t` are orders that arrive during period `t`.
- Sales: `s_t = min(d_t, I_t + A_t)` (demand is satisfied from on-hand
  inventory plus same-period arrivals).
- Unmet demand is lost (no backorders).

### Inventory Dynamics

```
I_{t+1} = max(0, I_t + A_t - d_t)
```

Ending inventory carries over to the next period and incurs holding cost.

### Objective

Maximize the cumulative reward over all periods:

```
Maximize  sum_t ( p * s_t  -  h * I_{t+1} )
```

where:

- `p` is the profit per unit sold.
- `h` is the holding cost per unit per period.
- `s_t` is units sold in period `t`.
- `I_{t+1}` is ending inventory after period `t`.

---

## VendingMachine Environment

The game is implemented as the `VendingMachine-v0` environment in the OR Agent
framework, following a two-player structure:

- **Player 0 (VM agent):** Controls ordering decisions.
- **Player 1 (Demand):** Provides customer demand each period.

### Basic Usage

```python
import or_agent as ta

# Create the environment
env = ta.make(env_id="VendingMachine-v0")

# Configure items
env.add_item(
    item_id="sku_001",
    description="Winter jacket",
    lead_time=4,
    profit=4,
    holding_cost=1
)

# Optionally add news events
env.add_news(day=5, news="Holiday sale: expect higher demand")

# Reset the environment
env.reset(num_players=2, num_days=50, initial_inventory_per_item=0)

# Game loop
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, step_info = env.step(action=action)

# Collect results
rewards, game_info = env.close()
print(f"Total Reward: {rewards[0]:.2f}")
```

### Environment Configuration

| Parameter                    | Description                              | Default |
|------------------------------|------------------------------------------|---------|
| `num_days`                   | Number of periods in the episode         | 10      |
| `initial_inventory_per_item` | Starting on-hand inventory per item      | 0       |
| `num_players`                | Always 2 (VM agent + Demand)             | 2       |

### Item Configuration

Each item added via `env.add_item()` requires:

| Parameter      | Type   | Description                               |
|----------------|--------|-------------------------------------------|
| `item_id`      | str    | Unique identifier for the product         |
| `description`  | str    | Human-readable product description        |
| `lead_time`    | int    | Periods until order arrives               |
| `profit`       | float  | Profit earned per unit sold               |
| `holding_cost` | float  | Cost per unit per period for holding      |

Items can be dynamically reconfigured each period via
`env.update_item_config()`, which supports changing lead time, profit, holding
cost, and description mid-game.

### News System

News events can be scheduled for specific days via `env.add_news(day, news)`.
The complete news schedule is visible to both agents from the start of the game.
News is purely informational -- it does not alter game rules, but agents can use
it to anticipate demand changes.

---

## Period Execution Sequence

Each period follows a strict four-phase execution order:

### Phase 1: VM Decision Phase

The VM agent (Player 0) receives an observation containing:

- Current period number and calendar date (if available).
- Product descriptions and cost parameters.
- On-hand inventory and in-transit summary.
- Complete order and demand history from previous periods.
- News schedule (with marker indicating the current day).

The agent responds with an ordering action.

### Phase 2: Arrival Resolution

Orders that are scheduled to arrive in this period are added to on-hand
inventory. The arrival information is recorded but not yet visible to the
agent -- it will appear in the period conclusion message.

### Phase 3: Demand Resolution

The Demand player (Player 1) provides customer demand for the period. Sales are
computed as `s_t = min(d_t, on_hand_inventory)`. Inventory is reduced
accordingly.

### Phase 4: Period Conclusion

The system generates a summary of the period's events:

- Orders that arrived (with original order date and actual lead time).
- Demand quantity, units sold, and units lost to stockout.
- Ending on-hand inventory.
- Period profit, holding cost, and net reward.

This summary becomes visible in the next period's observation.

**Important timing note:** When the agent makes decisions for period `N`, it can
see the conclusion from period `N-1`. Arrivals that occur during period `N` are
only visible starting from period `N+1`.

---

## Action Format

Actions are provided as JSON strings with an `action` field mapping item IDs to
order quantities:

```json
{
  "action": {
    "sku_001": 150,
    "sku_002": 75
  }
}
```

For strategies that include LLM reasoning, the JSON may also contain:

```json
{
  "rationale": "Step-by-step reasoning about the order decision...",
  "carry_over_insight": "Detected demand regime shift at period 10...",
  "action": {
    "sku_001": 150
  }
}
```

The environment parses the `action` field and ignores other fields.

---

## Observation Structure

The observation provided to the VM agent each period is a formatted text string
containing the following sections:

1. **Header:** Current period, total periods remaining.
2. **News schedule:** All scheduled news events, with a marker on the current
   day.
3. **Item summary:** For each item:
   - Description, profit per unit, holding cost per unit per period.
   - On-hand inventory and total in-transit units.
4. **History:** Chronological record of all previous periods, including:
   - Orders placed by the VM agent.
   - Period conclusion messages (arrivals, demand, sales, ending inventory,
     reward).

---

## Reward Structure

The reward is computed each period as:

```
R_t = p * s_t  -  h * I_{t+1}
```

where:

- `p * s_t` is revenue from sales.
- `h * I_{t+1}` is the holding cost on ending inventory.

The total episode reward is:

```
Total Reward = sum_{t=1}^{T} R_t
```

Only Player 0 (the VM agent) receives a reward. Player 1 (Demand) receives 0.

---

## Creating Custom Instances

Users can create custom benchmark instances by providing `train.csv` and
`test.csv` files in a directory.

### train.csv

Provides historical demand samples for algorithm initialization.

**Required columns:**

| Column                     | Description                                      |
|----------------------------|--------------------------------------------------|
| `exact_dates_{item_id}`    | Period identifier or calendar date               |
| `demand_{item_id}`         | Historical demand value for that period           |

**Example:**

```csv
exact_dates_mysku,demand_mysku
Period_1,100
Period_2,120
Period_3,95
Period_4,110
Period_5,105
```

### test.csv

Defines the evaluation scenario with demand, lead times, and cost parameters.

**Required columns:**

| Column                     | Description                                      |
|----------------------------|--------------------------------------------------|
| `exact_dates_{item_id}`    | Period identifier or calendar date               |
| `demand_{item_id}`         | Actual demand for the period                     |
| `lead_time_{item_id}`      | Lead time for orders placed this period (integer or `inf`) |
| `profit_{item_id}`         | Profit per unit sold                             |
| `holding_cost_{item_id}`   | Holding cost per unit per period                 |

**Optional columns:**

| Column                     | Description                                      |
|----------------------------|--------------------------------------------------|
| `description_{item_id}`    | Product description (used by LLM strategies)     |

**Example:**

```csv
exact_dates_mysku,demand_mysku,description_mysku,lead_time_mysku,profit_mysku,holding_cost_mysku
2025-01-06,100,Winter boots | Footwear,4,4,1
2025-01-13,120,Winter boots | Footwear,4,4,1
2025-01-20,95,Winter boots | Footwear,4,4,1
...
```

**Notes on lead time values:**

- Use integer values for deterministic lead times (e.g., `0`, `4`).
- Use `inf` to indicate a lost order (the order will never arrive).
- Different rows can have different lead time values to simulate stochastic
  supply.

### Running a Custom Instance

```bash
# OR strategy
python scripts/run_or.py \
  --demand-file custom_instance/test.csv \
  --real-instance-train custom_instance/train.csv \
  --promised-lead-time 4

# Full benchmark (all strategies)
python scripts/benchmark_all_strategies.py \
  --directory custom_instance/ \
  --model <model_name>
```

---

## Key Design Decisions

1. **Uncensored demand:** The agent always observes the true demand `d_t`,
   even when inventory is insufficient to fulfill it. This eliminates the
   censored-demand estimation problem and focuses evaluation on ordering policy
   quality.

2. **Zero initial inventory:** Episodes begin with no on-hand stock, forcing
   agents to build up inventory from the start.

3. **No backorders:** Unmet demand is lost. Customers do not wait for future
   fulfillment.

4. **Two-player structure:** Separating the VM agent from the Demand player
   allows demand to be driven by CSV files (for benchmarking) or by another
   LLM (for interactive play).

5. **Observation delay:** There is always a one-period delay between when
   events occur and when the agent can observe them in the conclusion message.
   This reflects realistic information delays in supply chain operations.
