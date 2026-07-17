"""Run the four fixed SGIRA decision chains on a frozen InventoryBench subset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sgira.executors import Route  # noqa: E402
from sgira.inventorybench import (  # noqa: E402
    load_inventorybench_instance,
    run_fixed_instance,
    save_fixed_run,
    summarize_runs,
)
from sgira.llm import OpenAICompatibleLLM  # noqa: E402


def read_instances(path: Path) -> list[Path]:
    values = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return [value if Path(value).is_absolute() else ROOT / "benchmark" / value for value in map(Path, values)]


def make_llm(model: str, output_dir: Path):
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return OpenAICompatibleLLM(
        model=model,
        cache_dir=str(output_dir / ".llm_cache"),
        audit_path=str(output_dir / "model_audit.jsonl"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-file", type=Path,
        default=ROOT / "sgira" / "smoke_set_12.txt",
    )
    parser.add_argument(
        "--routes", nargs="+", choices=[route.value for route in Route],
        default=[route.value for route in Route],
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sgira_runs" / "fixed_dev")
    parser.add_argument(
        "--model", default=os.getenv("MA_MODEL_GEMINI3_FLASH", "gemini-3-flash-preview")
    )
    args = parser.parse_args()

    if not args.instance_file.is_absolute():
        args.instance_file = ROOT / args.instance_file
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    instance_paths = read_instances(args.instance_file)
    needs_llm = any(route != Route.OR.value for route in args.routes)
    llm = make_llm(args.model, args.output_dir) if needs_llm else None
    runs = []
    for route_name in args.routes:
        route = Route(route_name)
        for index, instance_path in enumerate(instance_paths, 1):
            instance = load_inventorybench_instance(instance_path)
            relative = instance_path.relative_to(ROOT / "benchmark")
            target = args.output_dir / route.value / "results" / relative
            print(f"[{route.value} {index}/{len(instance_paths)}] {relative}", flush=True)
            result = run_fixed_instance(
                instance,
                route,
                llm=llm,
                decision_log=target / "decision_log.jsonl",
            )
            save_fixed_run(result, target)
            runs.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_runs(runs)
    summary.to_csv(args.output_dir / "scenario_summary.csv", index=False)
    overall_rows = []
    for route_name, group in pd.DataFrame([
        {"route": run.route, **run.metrics} for run in runs
    ]).groupby("route"):
        row = {"route": route_name}
        row.update({name: float(group[name].mean()) for name in group if name != "route"})
        overall_rows.append(row)
    (args.output_dir / "overall_summary.json").write_text(
        json.dumps(overall_rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Completed {len(runs)} runs. Summary: {args.output_dir}")


if __name__ == "__main__":
    main()
