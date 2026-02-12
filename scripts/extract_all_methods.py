"""
Extract decisions from all methods (OR, LLM, OR→LLM, LLM→OR) from empirical benchmark results.

This script:
1. Reads log files from empirical_results folders
2. Extracts order decisions for each method
3. Creates results.csv files in proper format
4. Evaluates all methods and generates evaluation JSON files
"""

import json
import re
from pathlib import Path
import pandas as pd
import subprocess
import sys


def extract_decisions_from_log(log_file_path, method_type):
    """
    Extract order decisions from a log file.

    Uses the most reliable extraction source: "=== Period X Summary ===" sections,
    which contain the actual executed order quantities. Falls back to VM Action/Decision
    JSON blocks if summaries are not found (handles alternate log formats).

    Args:
        log_file_path: Path to the log file
        method_type: Type of method ('or', 'llm', 'or_to_llm', 'llm_to_or')

    Returns:
        dict: {period: order_quantity}
    """
    if not log_file_path.exists():
        return {}

    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    decisions = {}

    # Extract from the most reliable source: "=== Period X Summary ===" sections
    # These contain the actual executed order quantities that occurred in that period.
    # Pattern: === Period X Summary ===\nitem_id: ordered=VALUE

    # First, find all period summaries to extract item IDs dynamically
    summary_pattern = r'=== Period (\d+) Summary ===\n([^:]+):\s*ordered=(\d+)'
    matches = re.findall(summary_pattern, content)

    for period_str, item_id, quantity in matches:
        period = int(period_str)
        if period not in decisions:  # Keep first occurrence (should be only one per period)
            decisions[period] = int(quantity)

    # If no matches found from summaries, fall back to VM Action JSON blocks
    # This handles any alternate log formats
    if not decisions:
        # Try both Decision and VM Action patterns (more flexible)
        pattern_fallback = r'Period (\d+) \([^)]+\).*?(?:Decision|VM Action):[^{]*\{[^{]*"action"\s*:\s*\{[^}]*"([^"]+)"\s*:\s*(\d+)'
        matches = re.findall(pattern_fallback, content, re.DOTALL)

        for period_str, item_id, quantity in matches:
            period = int(period_str)
            if period not in decisions:
                decisions[period] = int(quantity)

    return decisions


def process_instance(instance_dir, method_name, output_base_dir):
    """
    Process a single instance for a specific method.
    
    Args:
        instance_dir: Path to instance directory (e.g., archive/gemini-3-flash_bench/.../108775044)
        method_name: Method name ('or', 'llm', 'or_to_llm', 'llm_to_or')
        output_base_dir: Base output directory for results
    """
    # Get the log file for this method
    log_file = instance_dir / f"{method_name}_1.txt"
    
    if not log_file.exists():
        return None
    
    # Extract decisions
    decisions = extract_decisions_from_log(log_file, method_name)
    
    if not decisions:
        return None
    
    # Read test.csv to get the structure
    test_csv = instance_dir / "test.csv"
    if not test_csv.exists():
        return None
    
    test_df = pd.read_csv(test_csv)
    
    # Get item_id from column name
    demand_cols = [c for c in test_df.columns if c.startswith('demand_')]
    if not demand_cols:
        return None
    
    item_id = demand_cols[0].replace('demand_', '')
    
    # Create results DataFrame
    results_data = []
    for period in range(1, len(test_df) + 1):
        order = decisions.get(period, 0)
        results_data.append({
            'period': period,
            'order_quantity': order
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Get relative path from empirical_results/llm_bench folder (skip empirical_results and llm_bench parts)
    # e.g., empirical_results/gemini-3-flash_bench/real_trajectory/... -> real_trajectory/...
    parts = instance_dir.parts
    empirical_idx = parts.index('empirical_results')
    relative_parts = parts[empirical_idx + 2:]  # Skip 'empirical_results' and 'llm_bench'
    relative_path = Path(*relative_parts) if relative_parts else Path('.')
    
    # Create output directory
    output_dir = output_base_dir / relative_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    output_file = output_dir / "results.csv"
    results_df.to_csv(output_file, index=False)
    
    return output_file


def main():
    """Extract decisions from all methods in all archived benchmarks."""

    archive_dir = Path("empirical_results")
    results_dir = Path("results")

    # Define all LLMs and methods to process
    llms = [
        ("gemini-3-flash_bench", "gemini-3-flash", "Gemini 3 Flash"),
        ("gpt-5-mini_bench", "gpt-5-mini", "GPT-5 Mini"),
        ("grok-4.1-fast_bench", "grok-4.1-fast", "Grok 4.1 Fast"),
    ]

    # Method display names
    method_display = {
        "or": "OR",
        "llm": "LLM",
        "or_to_llm": "OR→LLM",
        "llm_to_or": "LLM→OR"
    }

    methods = ["or", "llm", "or_to_llm", "llm_to_or"]

    # Track OR folder for all models (they're all the same)
    or_processed = False
    or_output_dir = None
    
    for bench_folder, _, llm_display in llms:
        bench_path = archive_dir / bench_folder

        if not bench_path.exists():
            print(f"⚠️  Skipping {bench_folder} - not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {llm_display}")
        print(f"{'='*60}")

        for method in methods:
            print(f"\n📁 Extracting {method_display[method]} decisions...")

            # Special handling for OR: only process once and use unified folder
            if method == "or":
                if or_processed:
                    print(f"   (Skipping - OR results already extracted)")
                    continue
                or_processed = True
                # Use a unified OR folder instead of per-model
                output_dir = results_dir / "OR (capped base stock)" / "results"
                or_output_dir = output_dir
            else:
                # Create output directory using display names: "Model Name (Method)"
                spec_name = f"{llm_display} ({method_display[method]})"
                output_dir = results_dir / spec_name / "results"

            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all instance directories (use recursive search)
            instance_dirs = []
            for log_file in bench_path.rglob(f"{method}_1.txt"):
                instance_dir = log_file.parent
                instance_dirs.append(instance_dir)
            
            # Process each instance
            processed = 0
            for instance_dir in instance_dirs:
                result = process_instance(instance_dir, method, output_dir)
                if result:
                    processed += 1
            
            print(f"✅ Extracted {processed}/{len(instance_dirs)} instances")
            
            # Run evaluation
            if processed > 0:
                print(f"🔍 Evaluating {method_display[method]}...")

                # Determine the parent results directory for evaluation
                if method == "or":
                    spec_name = "OR (capped base stock)"
                else:
                    spec_name = f"{llm_display} ({method_display[method]})"

                spec_dir = results_dir / spec_name

                cmd = [
                    sys.executable,
                    "eval/evaluate_results.py",
                    "--benchmark-dir", "benchmark",
                    "--submission-dir", str(spec_dir)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    # Parse and display score
                    scores_path = spec_dir / "scores.json"
                    with open(scores_path) as f:
                        eval_data = json.load(f)
                        overall_score = eval_data.get("overall_score", 0)
                        print(f"📊 Overall score: {overall_score:.4f}")
                else:
                    print(f"❌ Evaluation failed: {result.stderr}")
    
    print(f"\n{'='*60}")
    print("✨ All extractions complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
