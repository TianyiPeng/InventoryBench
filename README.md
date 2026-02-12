# InventoryBench: LLM + Operations Research Benchmark

A comprehensive benchmark for evaluating how Large Language Models (LLMs) can collaborate with Operations Research (OR) algorithms on multi-period inventory control problems. InventoryBench provides **1,320 benchmark instances**, implementations of **4 agent strategies**, and a **VendingMachine** inventory game environment built on the [TextArena](https://github.com/LeonGuertler/TextArena) framework.

**Website & Leaderboard:** [https://tianyipeng.github.io/InventoryBench/](https://tianyipeng.github.io/InventoryBench/)

---

## Overview

Inventory management is a core operations research problem: deciding how much to order each period to balance holding costs against lost-sales penalties. This benchmark systematically evaluates four strategies for combining LLM reasoning with classical OR algorithms, measuring each against a clairvoyant (perfect-information) upper bound.

**Key metric:** Normalized Reward = agent\_reward / perfect\_score

---

## Strategies

InventoryBench supports four strategies for inventory ordering decisions. See [docs/strategies.md](docs/strategies.md) for full details.

| Strategy | Description |
|----------|-------------|
| **OR (Base-stock only)** | Pure statistical base-stock policy. No LLM. Uses empirical mean and standard deviation from historical demand to compute optimal order quantities. |
| **LLM-only** | The LLM makes all ordering decisions directly, given full game context: current inventory, demand history, product descriptions, and calendar dates. |
| **OR-to-LLM** | The OR algorithm computes a base-stock recommendation. The LLM sees this recommendation along with the full context and makes the final decision -- it may accept, adjust, or override the OR suggestion. |
| **LLM-to-OR** | The LLM analyzes the game state (demand history, product info, calendar) and proposes distribution parameters (mean, standard deviation, lead time) to the OR algorithm, which then computes the order quantity. |

---

## Benchmark

**1,320 total instances** spanning synthetic and real-world demand scenarios.

### Synthetic (720 instances)

Generated from 10 demand patterns with controlled variation:

- 10 demand patterns x 4 parameter variants x 2 realizations x 3 lead times x 3 cost ratios
- Demand patterns include stationary, trending, seasonal, and shifting distributions

### Real (600 instances)

Derived from H&M retail weekly sales data:

- 200 distinct product articles x 3 lead time settings
- Randomized profit-to-holding-cost ratios

### Settings

| Parameter | Values |
|-----------|--------|
| Lead time (L) | 0, 4, stochastic (drawn from {1, 2, 3, inf}) |
| Cost ratio (rho) | 0.50, 0.80, 0.95 |

### Instance format

Each instance is a directory containing two CSV files:

- `train.csv` -- Historical demand samples (context for the agent)
- `test.csv` -- Demand trajectory for evaluation (50 periods for synthetic, 48 for real)

---

## Quick Start

### 1. Install

Requires [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/TianyiPeng/InventoryBench.git
cd InventoryBench
uv sync
```

### 2. Set API keys

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your OpenAI / OpenRouter / other API keys
```

LLM-based strategies use OpenRouter by default. The OR-only strategy does not require an API key.

### 3. Run a single strategy on one instance

```bash
# OR baseline (no API key needed)
uv run python scripts/run_or.py \
  --demand-file benchmark/synthetic_trajectory/lead_time_0/p01_stationary_iid/v1_normal_100_25/r1_med/test.csv \
  --real-instance-train benchmark/synthetic_trajectory/lead_time_0/p01_stationary_iid/v1_normal_100_25/r1_med/train.csv \
  --promised-lead-time 0

# LLM-only (requires OPENROUTER_API_KEY in .env)
uv run python scripts/run_llm.py \
  --demand-file benchmark/synthetic_trajectory/lead_time_0/p01_stationary_iid/v1_normal_100_25/r1_med/test.csv \
  --real-instance-train benchmark/synthetic_trajectory/lead_time_0/p01_stationary_iid/v1_normal_100_25/r1_med/train.csv \
  --promised-lead-time 0 --model x-ai/grok-4.1-fast
```

Each strategy has its own script:

| Script | Strategy |
|--------|----------|
| `scripts/run_or.py` | OR (base-stock only) |
| `scripts/run_llm.py` | LLM-only |
| `scripts/run_or_to_llm.py` | OR-to-LLM |
| `scripts/run_llm_to_or.py` | LLM-to-OR |

### 4. Run all strategies on one instance

```bash
uv run python scripts/benchmark_all_strategies.py \
  --directory benchmark/synthetic_trajectory/lead_time_0/p01_stationary_iid/v1_normal_100_25/r1_med
```

This runs all four strategies plus the clairvoyant upper bound, saves results to `benchmark_results.json` in the instance directory, and prints a comparison table.

### 5. Run batch benchmark

```bash
uv run python scripts/run_batch_benchmark.py \
  --base-dir benchmark/synthetic_trajectory/lead_time_0
```

---

## Submitting Your Policy

Want to evaluate your own inventory control policy on InventoryBench? Here's the process:

**For detailed instructions, see [eval/README.md](eval/README.md).**

### Quick Overview

The standard submission structure is:
```
my_policy/
├── results/          # Instance-level results (auto-generated)
├── scores.json       # Evaluation summary (auto-generated)
└── README.md         # Your method description
```

Steps to create this:
1. **Implement Your Policy** — Create a class inheriting from `InventoryPolicy` in `eval/policy_template.py`
2. **Run Your Policy** — `python eval/run_baseline_policy.py --output-dir my_policy` → generates `my_policy/results/`
3. **Evaluate Results** — `python eval/evaluate_results.py --submission-dir my_policy` → generates `my_policy/scores.json`
4. **Add Documentation** — Create `README.md` describing your approach
5. **Submit** — Via GitHub PR or email to tianyipeng95@gmail.com

### Evaluation Metrics

Your policy is scored using **Normalized Reward**:

```
normalized_reward = max(0, agent_reward / perfect_foresight_reward)
```

Where:
- `agent_reward`: Total profit minus holding costs from your decisions
- `perfect_foresight_reward`: Optimal reward with perfect future demand knowledge

The score is the mean normalized reward across all 1,320 instances.

### Key Constraints

- You receive only `promised_lead_time` at initialization (0, 2, or 4 periods)
- You do NOT receive actual lead times during gameplay
- You must infer supply reliability from observed arrivals
- All other game context is fair game for your policy

**See [eval/README.md](eval/README.md) for comprehensive instructions, including the policy interface, evaluation framework, file formats, and submission guidelines.**

---

## Project Structure

```
InventoryBench/
  or_agent/                  Core package
    envs/                      VendingMachine environment
    agents/                    Agent implementations (LLM, OR, hybrid)
    wrappers/                  Observation and action wrappers
    core.py                    Base classes (Env, Agent, Wrapper)
    state.py                   Game state management
  scripts/
    run_or.py                  OR-only strategy
    run_llm.py                 LLM-only strategy
    run_or_to_llm.py           OR-to-LLM strategy
    run_llm_to_or.py           LLM-to-OR strategy
    benchmark_all_strategies.py  Run all strategies on an instance
    perfect_score.py           Clairvoyant upper bound computation
  benchmark/                   Benchmark instance data (train.csv + test.csv)
    synthetic_trajectory/        720 synthetic instances
    real_trajectory/             600 real (H&M) instances
  results/                   Benchmark results by model
    grok-4.1-fast_bench/       Grok 4.1 Fast results
    gpt-5-mini_bench/          GPT-5 Mini results
    gemini-3-flash_bench/      Gemini 3 Flash results
  leaderboard_website/       Community leaderboard for tracking AI agent performance
  docs/                      Detailed documentation
```

---

## Results

Pre-computed benchmark results for three LLMs are included in `results/`:

| Model | Directory |
|-------|-----------|
| Grok 4.1 Fast | `results/grok-4.1-fast_bench/` |
| GPT-5 Mini | `results/gpt-5-mini_bench/` |
| Gemini 3 Flash | `results/gemini-3-flash_bench/` |

Each results directory mirrors the `benchmark/` structure (`synthetic_trajectory/` and `real_trajectory/`) with `benchmark_results.json` files per instance.

When you run `benchmark_all_strategies.py` yourself, new results are saved as `benchmark_results.json` in the instance directory under `benchmark/`.

---

## Leaderboard

The `leaderboard_website/` directory contains a community leaderboard for tracking AI agent performance on the inventory control benchmark. It provides per-model, per-strategy breakdowns across all 1,320 instances.

To serve locally:

```bash
cd leaderboard_website
python serve.py
```

---

## Citation

If you use InventoryBench in your research, please cite:

```bibtex
@article{baek2025ai,
  title={AI Agents for Inventory Control: Human-LLM-OR Complementarity},
  author={Baek, Jackie and Fu, Yaopeng and Ma, Will and Peng, Tianyi},
  year={2025}
}
```

**Authors:** Jackie Baek*, Yaopeng Fu†, Will Ma‡, Tianyi Peng§

---

## Acknowledgments

This project builds on the [TextArena](https://github.com/LeonGuertler/TextArena) framework. We thank the TextArena team for their game infrastructure, which provides the foundational environment and agent interfaces used in InventoryBench.

---

## License

This project is licensed under the [MIT License](LICENSE).
