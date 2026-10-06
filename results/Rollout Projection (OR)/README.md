# Rollout Projection (OR)

**Overall Score**: 0.6330 (0.6330144033 with the unmodified `eval/evaluate_results.py`, all 1,320 instances)

A causal, non-LLM controller. Each period it fits an online forecast mixture to the demand observed so far,
keeps an exact belief over outstanding orders from observed arrivals, and picks an order by simulating
capped base-stock candidates over a 16-period horizon with 128 sampled demand/supply futures.

## Approach

- **Forecasts:** a bank of raw- and log-space level/trend/seasonal models with shared level uncertainty,
  weighted by online likelihood, plus causal category/calendar priors from the product description and dates.
- **Supply:** actual lead times are never observed. Outstanding orders are tracked with an exact belief over
  receipts; for `lead_time_stochastic` the rollout assumes the documented lead-time law
  (uniform over {1, 2, 3, inf}, i.e. 75% of orders arrive).
- **Decision:** for lead time 0, a deterministic quadrature of the forecast mixture gives the order;
  otherwise capped base-stock candidates are scored by Monte Carlo rollout (common random numbers, 128 paths,
  horizon 16) and the best one is ordered. A Numba kernel runs the rollout.
- No LLM, no file or network access, no future demand or hidden lead times at decision time.

Several components are independent implementations inspired by
[SabreAgent](https://arxiv.org/abs/2609.19760) (forecast bank, surplus clearance, capped-policy search).
This is not the SabreAgent authors' code.

## Results

| Cell | This policy | SabreAgent (paper, Table 1) |
|---|---:|---:|
| real / L=0 | 0.7950 | 0.8018 |
| real / L=4 | 0.4401 | 0.4447 |
| real / stochastic | 0.4710 | 0.4714 |
| synthetic / L=0 | 0.8675 | 0.8649 |
| synthetic / L=4 | 0.6773 | 0.6685 |
| synthetic / stochastic | 0.5150 | 0.5063 |
| **overall** | **0.6330** | **0.6311** |

SabreAgent numbers are copied from the paper; its code was not run here. The best entry currently in this
repository's `results/` is Gemini 3 Flash (OR→LLM) at 0.5380.

## Reproduce

```bash
git lfs pull
pip install -r "results/Rollout Projection (OR)/requirements.txt"
python "results/Rollout Projection (OR)/run.py" --output-dir my_run
python eval/evaluate_results.py --submission-dir my_run
```

`run.py` only swaps `ExamplePolicy` for `InventoryBenchPolicy` and calls the unmodified
`eval/run_baseline_policy.py`. The policy is deterministic (fixed seeds); a rerun reproduces every
order in `results/` exactly. Runtime is about 4 minutes on one CPU core.

## Limitations (please read)

- **Not a blind test.** The policy was developed over many iterations on these same 1,320 public instances.
  The final setting (128 rollout paths, chosen over 16/32/64) was selected on freshly generated cases only,
  but earlier design choices were informed by benchmark scores. Treat 0.6330 as a development-set result.
- **Small margin, mixed confirmation.** Going from 32 to 128 rollout paths gains +0.0005 on the benchmark,
  but -0.0004 on 288 fresh generated cases (paired 95% bootstrap [-0.0026, +0.0016]). With simulated supply
  survival of 0.50 / 0.75 / 0.95 the change is +0.0076 / -0.0004 / -0.0015. The rollout always assumes 75% survival.
- **Real-data cells** are slightly below SabreAgent's reported values; the overall lead comes from the synthetic cells.
- **Input-type sensitivity.** Orders can differ when equal-valued demands are passed as `int` versus `float`
  in `initial_samples`. The scores above use the stock runner, which passes the CSV values as read by pandas.
  A variant that converts all demand inputs to float scores 0.63283 overall.
- Requires `numba`, which is not in the project dependencies.
