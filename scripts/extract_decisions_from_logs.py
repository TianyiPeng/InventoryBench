#!/usr/bin/env python3
"""
Extract inventory decisions from agent log files (or_1.txt, llm_1.txt, etc.)
and create results.csv files for evaluation.
"""

import json
import re
import argparse
from pathlib import Path
import pandas as pd


def extract_decisions_from_log(log_file: Path, item_id: str):
    """
    Extract period-by-period decisions from a log file.

    FIXED v2: Extracts the ACTUAL executed order from "=== Period X Summary ==="
    sections, which contain the final order placed for each period.

    The period summary format:
        === Period X Summary ===
        item_id: ordered=VALUE, arrived=..., ...

    This avoids the issue of multiple decision JSON blocks (OR recommendations,
    LLM decisions, Hybrid decisions) per period by using the authoritative
    period summary that shows what actually happened.

    Returns:
        List of (period, order_quantity) tuples
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    decisions = []

    # Extract from period summaries: "=== Period X Summary ===" sections
    # Pattern: === Period X Summary === ... item_id: ordered=VALUE
    summary_pattern = r'=== Period (\d+) Summary ===\n.*?{0}:.*?ordered=(\d+)'.format(re.escape(item_id))

    for match in re.finditer(summary_pattern, content, re.DOTALL):
        period = int(match.group(1))
        ordered_qty = float(match.group(2))
        decisions.append((period, ordered_qty))

    # If no summaries found, fall back to Hybrid Decision blocks
    # (for different log formats)
    if not decisions:
        hybrid_pattern = r'Period (\d+) \([^)]+\) Hybrid Decision:[^{]*\{[^}]*"action"[^}]*\{[^}]*"({0})":\s*(\d+(?:\.\d+)?)'.format(re.escape(item_id))
        for match in re.finditer(hybrid_pattern, content, re.DOTALL):
            period = int(match.group(1))
            order_qty = float(match.group(2))
            decisions.append((period, order_qty))

    # Sort by period
    decisions.sort(key=lambda x: x[0])

    return decisions


def process_agent_logs(results_dir: Path, agent_name: str, output_suffix: str = "_decisions"):
    """
    Process all log files for a specific agent across all batches.
    
    Args:
        results_dir: Path to results directory (e.g., results/gemini-3-flash_bench)
        agent_name: Agent type (e.g., 'or', 'llm', 'llm_to_or', 'or_to_llm')
        output_suffix: Suffix to add to output directory name
    """
    log_filename = f"{agent_name}_1.txt"
    
    processed_count = 0
    error_count = 0
    
    # Find all log files
    log_files = list(results_dir.rglob(log_filename))
    
    print(f"Found {len(log_files)} {log_filename} files")
    
    for log_file in log_files:
        try:
            # Get instance directory
            instance_dir = log_file.parent
            
            # Get item_id from train.csv
            train_csv = instance_dir / "train.csv"
            if not train_csv.exists():
                print(f"  Warning: train.csv not found in {instance_dir.relative_to(results_dir)}")
                error_count += 1
                continue
            
            # Read CSV header to get item_id
            with open(train_csv) as f:
                header = f.readline().strip()
            
            # Extract item_id from column name like "demand_chips(Regular)" or "demand_108775044"
            if ',' in header:
                demand_col = header.split(',')[1]  # Second column is demand
                if demand_col.startswith('demand_'):
                    item_id = demand_col[7:]  # Remove 'demand_' prefix
                else:
                    # Fallback: use directory name
                    item_id = instance_dir.name
            else:
                item_id = instance_dir.name
            
            # Extract decisions
            decisions = extract_decisions_from_log(log_file, item_id)
            
            if not decisions:
                print(f"  Warning: No decisions found in {log_file.relative_to(results_dir)}")
                error_count += 1
                continue
            
            # Create DataFrame
            df = pd.DataFrame(decisions, columns=['period', 'order_quantity'])
            
            # Create output directory
            relative_path = instance_dir.relative_to(results_dir)
            output_dir = results_dir.parent / f"{results_dir.name}{output_suffix}" / relative_path
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            output_file = output_dir / "results.csv"
            df.to_csv(output_file, index=False)
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} instances...")
                
        except Exception as e:
            print(f"  Error processing {log_file.relative_to(results_dir)}: {e}")
            error_count += 1
    
    print(f"\nCompleted: {processed_count} instances processed, {error_count} errors")
    print(f"Output directory: {results_dir.parent / f'{results_dir.name}{output_suffix}'}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract inventory decisions from agent log files'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Path to results directory (e.g., results/gemini-3-flash_bench)'
    )
    parser.add_argument(
        '--agent',
        type=str,
        required=True,
        choices=['or', 'llm', 'llm_to_or', 'or_to_llm'],
        help='Agent type to extract decisions from'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='_decisions',
        help='Suffix to add to output directory name (default: _decisions)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return
    
    print(f"{'='*70}")
    print(f"Extracting {args.agent} decisions from {results_dir}")
    print(f"{'='*70}")
    
    process_agent_logs(results_dir, args.agent, args.output_suffix)


if __name__ == '__main__':
    main()
