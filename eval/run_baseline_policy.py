"""
Run Baseline Policy on InventoryBench

This script:
1. Enumerates all instances in benchmark/ folder
2. Runs YesterdayDemandPolicy on each instance
3. Outputs results/ folder with same structure as benchmark/
4. Each instance gets results.csv with columns: period, order_quantity

Usage:
    python eval/run_baseline_policy.py --benchmark-dir benchmark --output-dir results/yesterday_demand
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add parent directory to path to import policy_template
sys.path.insert(0, str(Path(__file__).parent))
from policy_template import YesterdayDemandPolicy


def detect_promised_lead_time(instance_path: str) -> int:
    """
    Auto-detect promised lead time from folder path.
    
    Returns:
        - 0 if 'lead_time_0' found in path
        - 4 if 'lead_time_4' found in path
        - 2 if 'lead_time_stochastic' found in path
    """
    path_str = str(instance_path).replace('\\', '/')
    
    if 'lead_time_0/' in path_str or '/lead_time_0' in path_str or path_str.endswith('lead_time_0'):
        return 0
    elif 'lead_time_4/' in path_str or '/lead_time_4' in path_str or path_str.endswith('lead_time_4'):
        return 4
    elif 'lead_time_stochastic/' in path_str or '/lead_time_stochastic' in path_str:
        return 2
    
    raise ValueError(f"Could not detect promised lead time from path: {instance_path}")


def load_instance(instance_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Load train.csv and test.csv for an instance.
    
    Returns:
        (train_df, test_df, item_id)
    """
    train_path = instance_dir / "train.csv"
    test_path = instance_dir / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise ValueError(f"Missing train.csv or test.csv in {instance_dir}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Extract item_id from column names (format: demand_{item_id})
    demand_cols = [col for col in test_df.columns if col.startswith('demand_')]
    if not demand_cols:
        raise ValueError(f"No demand columns found in {test_path}")
    
    item_id = demand_cols[0][len('demand_'):]
    
    return train_df, test_df, item_id


def simulate_instance(
    policy,
    test_df: pd.DataFrame,
    item_id: str
) -> pd.DataFrame:
    """
    Simulate policy on instance with full state transitions.
    
    Args:
        policy: InventoryPolicy instance
        test_df: Test data with demand trajectory
        item_id: SKU identifier
    
    Returns:
        results_df: DataFrame with columns [period, order_quantity]
    """
    num_periods = len(test_df)
    results = []
    
    # State tracking
    on_hand_inventory = 0.0
    in_transit_orders = {}  # {arrival_period: quantity}
    
    previous_demand = 0.0
    previous_order = 0.0
    previous_arrivals = 0.0
    
    for period_idx in range(num_periods):
        period = period_idx + 1
        row = test_df.iloc[period_idx]
        
        # Extract metadata for current period
        current_date = str(row.get(f'exact_dates_{item_id}', f'Period_{period}'))
        profit_per_unit = float(row[f'profit_{item_id}'])
        holding_cost_per_unit = float(row[f'holding_cost_{item_id}'])
        actual_lead_time = row[f'lead_time_{item_id}']
        
        # Handle infinite lead time (lost orders)
        if isinstance(actual_lead_time, str) and actual_lead_time.lower() == 'inf':
            actual_lead_time = float('inf')
        elif np.isinf(float(actual_lead_time)):
            actual_lead_time = float('inf')
        else:
            actual_lead_time = int(actual_lead_time)
        
        # === PERIOD EXECUTION SEQUENCE ===
        
        # 1. DECISION PHASE: Policy decides order
        in_transit_total = sum(in_transit_orders.values())
        
        order_quantity = policy.get_order(
            period=period,
            current_date=current_date,
            on_hand_inventory=on_hand_inventory,
            in_transit_total=in_transit_total,
            previous_demand=previous_demand,
            previous_order=previous_order,
            previous_arrivals=previous_arrivals,
            profit_per_unit=profit_per_unit,
            holding_cost_per_unit=holding_cost_per_unit
        )
        
        # Ensure non-negative integer
        order_quantity = max(0, int(order_quantity))
        
        # Record decision
        results.append({
            'period': period,
            'order_quantity': order_quantity
        })
        
        # Schedule arrival (if not lost)
        if not np.isinf(actual_lead_time):
            arrival_period = period + actual_lead_time
            in_transit_orders[arrival_period] = in_transit_orders.get(arrival_period, 0) + order_quantity
        # If lead_time=inf, order is lost (never added to in_transit_orders)
        
        # 2. ARRIVAL RESOLUTION: Process arrivals for this period
        arrivals = in_transit_orders.pop(period, 0)
        on_hand_inventory += arrivals
        
        # 3. DEMAND RESOLUTION: Satisfy demand from on-hand
        actual_demand = float(row[f'demand_{item_id}'])
        units_sold = min(actual_demand, on_hand_inventory)
        on_hand_inventory -= units_sold
        
        # Update previous state for next period
        previous_demand = actual_demand
        previous_order = order_quantity
        previous_arrivals = arrivals
    
    return pd.DataFrame(results)


def run_all_instances(benchmark_dir: Path, output_dir: Path):
    """
    Enumerate all instances in benchmark/ and run policy.
    
    Creates output_dir with same structure as benchmark_dir, but with
    results.csv instead of train.csv and test.csv.
    """
    # Find all instance directories (those containing test.csv)
    instance_dirs = []
    for root, dirs, files in os.walk(benchmark_dir):
        if 'test.csv' in files and 'train.csv' in files:
            instance_dirs.append(Path(root))
    
    print(f"Found {len(instance_dirs)} instances in {benchmark_dir}")
    
    # Process each instance
    for idx, instance_dir in enumerate(instance_dirs, 1):
        # Compute relative path from benchmark_dir
        rel_path = instance_dir.relative_to(benchmark_dir)
        
        print(f"\n[{idx}/{len(instance_dirs)}] Processing: {rel_path}")
        
        try:
            # Load instance data
            train_df, test_df, item_id = load_instance(instance_dir)
            
            # Extract initial samples from train.csv
            date_col = f'exact_dates_{item_id}'
            demand_col = f'demand_{item_id}'
            
            if date_col not in train_df.columns or demand_col not in train_df.columns:
                print(f"  ⚠️  Skipping: Missing required columns in train.csv")
                continue
            
            train_dates = train_df[date_col].tolist()
            train_demands = train_df[demand_col].tolist()
            initial_samples = list(zip(train_dates, train_demands))
            
            # Detect promised lead time from path
            promised_lead_time = detect_promised_lead_time(instance_dir)
            
            # Get cost parameters from first row of test.csv
            first_row = test_df.iloc[0]
            profit_per_unit = float(first_row[f'profit_{item_id}'])
            holding_cost_per_unit = float(first_row[f'holding_cost_{item_id}'])
            
            # Extract product description if available (real trajectories have this)
            desc_col = f'description_{item_id}'
            product_description = str(first_row[desc_col]) if desc_col in first_row else None
            
            # Initialize policy
            policy = YesterdayDemandPolicy(
                item_id=item_id,
                initial_samples=initial_samples,
                promised_lead_time=promised_lead_time,
                profit_per_unit=profit_per_unit,
                holding_cost_per_unit=holding_cost_per_unit,
                product_description=product_description
            )
            
            # Run simulation
            results_df = simulate_instance(policy, test_df, item_id)
            
            # Save results
            output_instance_dir = output_dir / rel_path
            output_instance_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_instance_dir / "results.csv"
            results_df.to_csv(output_path, index=False)
            
            print(f"  ✓ Saved: {output_path}")
            print(f"    Periods: {len(results_df)}, Total orders: {results_df['order_quantity'].sum():.0f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Completed! Results saved to: {output_dir}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description='Run baseline policy on InventoryBench instances'
    )
    parser.add_argument(
        '--benchmark-dir',
        type=str,
        default='benchmark',
        help='Path to benchmark parent directory (will auto-detect subfolders)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='eval/example_policy_test',
        help='Path to output parent directory (will mirror benchmark structure)'
    )
    
    args = parser.parse_args()
    
    benchmark_parent = Path(args.benchmark_dir)
    output_parent = Path(args.output_dir)
    
    if not benchmark_parent.exists():
        print(f"Error: Benchmark directory not found: {benchmark_parent}")
        sys.exit(1)
    
    # Auto-detect benchmark subfolders
    benchmark_subdirs = []
    for trajectory_type in ['real_trajectory', 'synthetic_trajectory']:
        for lead_time in ['lead_time_0', 'lead_time_4', 'lead_time_stochastic']:
            subdir = benchmark_parent / trajectory_type / lead_time
            if subdir.exists():
                benchmark_subdirs.append((subdir, trajectory_type, lead_time))
    
    if not benchmark_subdirs:
        print(f"Error: No benchmark subfolders found in {benchmark_parent}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"Running YesterdayDemandPolicy on InventoryBench")
    print(f"{'='*70}")
    print(f"Found {len(benchmark_subdirs)} batches to process")
    print(f"{'='*70}")
    
    # Process each batch
    for benchmark_dir, trajectory_type, lead_time in benchmark_subdirs:
        output_dir = output_parent / trajectory_type / lead_time
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Processing: {trajectory_type}/{lead_time}")
        print(f"Benchmark: {benchmark_dir}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}")
        
        run_all_instances(benchmark_dir, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ All batches completed! Results saved to: {output_parent}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
