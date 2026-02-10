"""
Build leaderboard statistics from benchmark results.

This script aggregates benchmark_results.json files from *_bench directories
and generates leaderboard data files (YAML and JSON) for the benchmark website.

Usage:
    python leaderboard_website/build_stats.py

Output:
    - leaderboard_website/data/leaderboard.json (for static HTML site)
"""
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent.parent  # repo root
RESULTS_DIR = ROOT / "results"
WEBSITE_DIR = Path(__file__).resolve().parent  # benchmark_website directory
JSON_DATA_DIR = WEBSITE_DIR / "data"  # For static HTML site


BENCH_DIR_SUFFIX = "_bench"

# Human‑readable labels for LLMs and methods
LLM_LABELS = {
    "gemini-3-flash_bench": "Gemini 3 Flash",
    "gpt-5-mini_bench": "GPT‑5 Mini",
    "grok-4.1-fast_bench": "Grok 4.1 Fast",
}

METHOD_LABELS = {
    "or": "OR",
    "llm": "LLM",
    "llm_to_or": "LLM→OR",
    "or_to_llm": "OR→LLM",
}


def iter_bench_dirs():
    """Iterate over all benchmark directories (*_bench) in results/."""
    for child in RESULTS_DIR.iterdir():
        if child.is_dir() and child.name.endswith(BENCH_DIR_SUFFIX):
            yield child


def parse_family_and_lead(instance_dir: str):
    """
    Parse dataset family (synthetic/real) and lead_time_x from instance_dir string.
    Falls back to None if pattern not found.
    """
    parts = instance_dir.replace("\\", "/").split("/")
    family = None
    lead = None
    for i, p in enumerate(parts):
        if p in ("synthetic_trajectory", "real_trajectory"):
            family = p
        if p.startswith("lead_time_"):
            lead = p
    return family, lead


def aggregate_for_bench(bench_dir: Path):
    """
    Aggregate ratio_to_perfect across all benchmark_results.json files
    in a single *_bench directory, per method.
    """
    bench_name = bench_dir.name
    llm_id = bench_name.replace(BENCH_DIR_SUFFIX, "")

    all_methods = {}

    json_paths = list(bench_dir.rglob("benchmark_results.json"))
    if not json_paths:
        return []

    model_name = None

    for path in json_paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if model_name is None:
            model_name = data.get("model")

        instance_dir = data.get("instance_dir", "")
        family, lead = parse_family_and_lead(instance_dir)

        results = data.get("results", {})
        for method, res in results.items():
            if method == "perfect_score":
                continue
            ratio = res.get("ratio_to_perfect")
            if ratio is None:
                continue
            # Truncate negative ratios to 0 before averaging (Section 4.2: prevent
            # outliers, e.g. one bad run with large negative profit, from having
            # outsize impact on average performance).
            ratio_truncated = max(0.0, ratio)

            m = all_methods.setdefault(
                method,
                {
                    "ratios": [],
                    "by_family": {},
                    "by_lead": {},
                },
            )
            m["ratios"].append(ratio_truncated)

            if family:
                fam_bucket = m["by_family"].setdefault(family, [])
                fam_bucket.append(ratio_truncated)
            if lead:
                lead_bucket = m["by_lead"].setdefault(lead, [])
                lead_bucket.append(ratio_truncated)

    rows = []
    for method, agg in all_methods.items():
        ratios = agg["ratios"]
        if not ratios:
            continue

        row = {
            "llm_id": llm_id,
            "llm_label": LLM_LABELS.get(bench_name, bench_name),
            "bench_dir": bench_name,
            "model_name": model_name,
            "method_id": method,
            "method_label": METHOD_LABELS.get(method, method),
            "mean_ratio": float(mean(ratios)),
        }

        # Per‑family means
        by_family_out = {}
        for fam, vals in agg["by_family"].items():
            by_family_out[fam] = float(mean(vals))
        if by_family_out:
            row["by_family_mean_ratio"] = by_family_out

        # Per‑lead‑time means
        by_lead_out = {}
        for lead, vals in agg["by_lead"].items():
            by_lead_out[lead] = float(mean(vals))
        if by_lead_out:
            row["by_lead_mean_ratio"] = by_lead_out

        rows.append(row)

    return rows


def main():
    """Main entry point: aggregate all benchmarks and write output files."""
    all_rows = []
    for bench_dir in iter_bench_dirs():
        all_rows.extend(aggregate_for_bench(bench_dir))

    # Sort descending by mean_ratio
    all_rows.sort(key=lambda r: r["mean_ratio"], reverse=True)

    # Write JSON for static HTML site
    JSON_DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = JSON_DATA_DIR / "leaderboard.json"
    json_data = {"methods": all_rows}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {json_path} with {len(all_rows)} methods.")


if __name__ == "__main__":
    main()
