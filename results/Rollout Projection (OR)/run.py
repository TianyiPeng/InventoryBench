"""Run this policy on all 1,320 instances with the unmodified eval/run_baseline_policy.py loop.

Usage (from the repository root, after `git lfs pull`):
    python "results/Rollout Projection (OR)/run.py" --output-dir my_run
    python eval/evaluate_results.py --submission-dir my_run
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(HERE))

import run_baseline_policy as runner  # noqa: E402
from policy import InventoryBenchPolicy  # noqa: E402

# The documented one-line swap: replace ExamplePolicy with this policy class.
runner.ExamplePolicy = InventoryBenchPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default=str(ROOT / "benchmark"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0], "--benchmark-dir", args.benchmark_dir, "--output-dir", args.output_dir]
    runner.main()
