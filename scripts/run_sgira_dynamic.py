"""Run Rule-Router and SGIRA on a frozen InventoryBench subset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sgira.controller import ControllerConfig  # noqa: E402
from sgira.inventorybench import (  # noqa: E402
    load_inventorybench_instance,
    run_routed_instance,
    save_routed_run,
    summarize_routed_runs,
)
from sgira.llm import OpenAICompatibleLLM  # noqa: E402


def read_instances(path: Path) -> list[Path]:
    values = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return [
        value if value.is_absolute() else ROOT / "benchmark" / value
        for value in map(Path, values)
    ]


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


def controller_config(ablation: str) -> ControllerConfig:
    return ControllerConfig(
        include_context=ablation != "no_context",
        include_memory=ablation != "no_memory",
        include_supply_diagnosis=ablation != "no_supply_diagnosis",
    )


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-file", type=Path,
        default=ROOT / "sgira" / "smoke_set_12.txt",
    )
    parser.add_argument(
        "--methods", nargs="+", choices=("RULE_ROUTER", "SGIRA"),
        default=("RULE_ROUTER", "SGIRA"),
    )
    parser.add_argument(
        "--ablation",
        choices=("none", "no_context", "no_memory", "no_supply_diagnosis"),
        default="none",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "sgira_runs" / "dynamic_dev"
    )
    parser.add_argument(
        "--model", default=os.getenv(
            "MA_MODEL_GEMINI3_FLASH", "gemini-3-flash-preview"
        )
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Replay completed periods from decision logs and continue safely.",
    )
    args = parser.parse_args()
    if not args.instance_file.is_absolute():
        args.instance_file = ROOT / args.instance_file
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    paths = read_instances(args.instance_file)
    llm = make_llm(args.model, args.output_dir)
    runs = []
    for method in args.methods:
        for index, path in enumerate(paths, 1):
            relative = path.relative_to(ROOT / "benchmark")
            target = args.output_dir / method / "results" / relative
            print(f"[{method} {index}/{len(paths)}] {relative}", flush=True)
            run = run_routed_instance(
                load_inventorybench_instance(path),
                method,
                llm=llm,
                controller_config=controller_config(args.ablation),
                decision_log=target / "decision_log.jsonl",
                resume=args.resume,
                progress=lambda period, horizon, replayed: print(
                    f"  period {period}/{horizon} "
                    f"({'replayed' if replayed else 'saved'})",
                    flush=True,
                ),
            )
            save_routed_run(run, target)
            runs.append(run)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarize_routed_runs(runs).to_csv(
        args.output_dir / "scenario_summary.csv", index=False
    )
    rows = pd.DataFrame([
        {"method": run.method, **run.metrics} for run in runs
    ])
    overall = []
    for method, group in rows.groupby("method"):
        summary = {"method": method}
        summary.update({
            name: float(group[name].mean()) for name in group if name != "method"
        })
        overall.append(summary)
    (args.output_dir / "overall_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps({
            "model": args.model,
            "methods": args.methods,
            "ablation": args.ablation,
            "resume": args.resume,
            "instance_file": str(args.instance_file),
        }, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Completed {len(runs)} dynamic runs. Summary: {args.output_dir}")


if __name__ == "__main__":
    main()
