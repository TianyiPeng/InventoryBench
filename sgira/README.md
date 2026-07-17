# SGIRA inventory routing implementation

This package implements the research handoff milestones M1--M5 without changing
the repository's earlier multi-agent experiments.

## Decision methods

The four fixed executors are shared by every baseline and router:

- `OR`: capped base-stock policy.
- `LLM`: the model directly proposes an order.
- `OR_TO_LLM`: OR proposes an order and the model audits it; adjustments are
  bounded to +/-30% and remain subject to the common order cap.
- `LLM_TO_OR`: the model estimates protection-period moments and OR converts
  those moments into an order.

`RULE_ROUTER` and `SGIRA` call those same executor objects. They do not contain
separate copies of the ordering algorithms.

## M5 routing design

Each pre-decision observation is transformed into:

- recent demand mean and sample standard deviation;
- coefficient of variation (`CV`);
- normalized linear trend;
- normalized adjacent-window mean shift;
- protection-period inventory coverage;
- oldest-order delay score and overdue-order count.

The frozen first-version human rules use these thresholds:

- `abs(trend) >= 0.05`;
- `abs(shift) >= 0.20`;
- `CV >= 0.75`;
- `DelayScore >= 1.50`.

Stable demand and normal supply select `OR`; demand anomalies select
`LLM_TO_OR`; supply anomalies select `OR_TO_LLM`; simultaneous demand and
supply anomalies select `LLM`.

The Gemini controller returns only a route, confidence, regimes, reason codes,
a short reason, and an optional memory update. It never returns an order. A
confidence below `0.5` falls back to the human rules. Invalid JSON, an invalid
route, or an API failure falls back to `OR`. Memory contains at most three
200-character entries, each with a generation period and three-period TTL.

The controller costs one model call per period. A selected non-OR route causes
one additional executor call. Prompts, inputs, raw outputs, parsed outputs,
corrections, fallback reasons, route switches, route shares, and call counts are
written to the decision and metric logs.

## Leakage controls

Policy inputs never include current/future demand, future arrival periods,
actual future lead times, loss outcomes, synthetic scenario labels, or another
method's results. Tests exercise these restrictions on complete
InventoryBench trajectories.

## Commands

Run all offline tests:

```bash
python -m unittest discover -s tests -p "test_sgira_*.py" -v
```

Run the four fixed methods on the frozen 12-instance smoke set:

```bash
python scripts/run_sgira_fixed.py \
  --instance-file sgira/smoke_set_12.txt \
  --routes OR LLM OR_TO_LLM LLM_TO_OR \
  --output-dir sgira_runs/fixed_gemini_smoke \
  --model gemini-3-flash-preview
```

Run Rule-Router and SGIRA on the same set:

```bash
python scripts/run_sgira_dynamic.py \
  --instance-file sgira/smoke_set_12.txt \
  --methods RULE_ROUTER SGIRA \
  --output-dir sgira_runs/dynamic_gemini_smoke \
  --model gemini-3-flash-preview \
  --resume
```

Each completed period is appended to `decision_log.jsonl`, and model responses
are cached on disk. If the process is interrupted, rerun the exact same command
with `--resume`: saved orders are replayed locally, controller memory is
restored, and API calls continue from the first unfinished period. Omitting
`--resume` intentionally starts a fresh run and clears the old decision log.

Both model-backed commands make real API calls through an OpenAI-compatible
endpoint. Create a local `.env` (never commit it) with:

```dotenv
MA_LLM_BASE_URL=https://company-gateway.example/v1
MA_LLM_API_KEY=replace-with-your-company-key
MA_LLM_MODEL=gemini-3-flash-preview
MA_MODEL_GEMINI3_FLASH=gemini-3-flash-preview
MA_LLM_JSON_MODE=true
MA_LLM_TRUST_ENV=true
```

The exact URL and model identifier are gateway-specific. If the endpoint does
not implement OpenAI JSON mode, set `MA_LLM_JSON_MODE=false`; SGIRA still
extracts and validates the returned JSON object.
Set `MA_LLM_TRUST_ENV=false` only when inherited proxy variables should be
ignored (for example, an unavailable local SOCKS proxy on a remote machine).
