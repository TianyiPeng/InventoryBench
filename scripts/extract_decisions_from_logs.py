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
    
    Supports two formats:
    1. OR/hybrid agents: Period X (...) Decision ... "action": {"item_id": value}
    2. LLM agents: Period X (...) VM Action ... "action": {"item_id": value}
    
    Returns:
        List of (period, order_quantity) tuples
    """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    decisions = []
    
    # Try OR/hybrid format first: "Decision" keyword
    # Pattern: Period X (...) Decision ... { ... "action" ... { "item_id": value } ... }
    pattern_or = r'Period (\d+) \([^)]+\)[^\{]*Decision[^\{]*\{[^}]*"action"[^}]*\{[^}]*"([^"]+)":\s*(\d+(?:\.\d+)?)'
    matches = re.finditer(pattern_or, content, re.DOTALL)
    
    for match in matches:
        period = int(match.group(1))
        matched_item_id = match.group(2)
        order_qty = float(match.group(3))
        
        if matched_item_id == item_id:
            decisions.append((period, order_qty))
    
    # If no matches, try LLM format: "VM Action" keyword with nested JSON
    if not decisions:
        # Pattern: Period X (...) VM Action: ... { ... "action": { "item_id": value } ... }
        pattern_llm = r'Period (\d+) \([^)]+\)\s+VM Action:[^{]*\{[^}]*"action"[^}]*\{[^}]*"([^"]+)":\s*(\d+(?:\.\d+)?)'
        matches = re.finditer(pattern_llm, content, re.DOTALL)
        
        for match in matches:
            period = int(match.group(1))
            matched_item_id = match.group(2)
            order_qty = float(match.group(3))
            
            if matched_item_id == item_id:
                decisions.append((period, order_qty))
    
    # Remove duplicates (keep first occurrence of each period)
    seen_periods = set()
    unique_decisions = []
    for period, qty in decisions:
        if period not in seen_periods:
            unique_decisions.append((period, qty))
            seen_periods.add(period)
    
    # Sort by period
    unique_decisions.sort(key=lambda x: x[0])
    
    return unique_decisions


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
