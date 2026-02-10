"""
Perfect Score Calculator for Vending Machine Environment.

This script calculates the theoretical maximum profit achievable:
  Perfect Score = sum(all demand) × profit_per_unit

This is a simple upper bound that does NOT consider:
- Lead time (all demand is counted)
- Holding costs
- Initial inventory constraints

This provides a consistent benchmark across all lead time configurations.

Usage:
  python perfect_score.py --demand-file path/to/test.csv
"""

import os
import sys
import argparse
import pandas as pd

# Fix stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def extract_item_ids(df: pd.DataFrame) -> list:
    """Extract item IDs from CSV columns (columns starting with 'demand_')."""
    item_ids = []
    for col in df.columns:
        if col.startswith('demand_'):
            item_id = col.replace('demand_', '')
            item_ids.append(item_id)
    return item_ids


def calculate_perfect_score(csv_path: str) -> dict:
    """
    Calculate the perfect score for a given test CSV file.
    
    Perfect Score = sum(all demand) × profit_per_unit
    
    This is a simple upper bound that ignores lead time and holding costs.
    
    Args:
        csv_path: Path to the test.csv file
        
    Returns:
        Dictionary containing perfect score details for each item
    """
    df = pd.read_csv(csv_path)
    item_ids = extract_item_ids(df)
    
    if not item_ids:
        raise ValueError("No item columns found in CSV. Expected columns like 'demand_<item_id>'")
    
    results = {}
    total_perfect_score = 0
    
    for item_id in item_ids:
        # Extract columns for this item
        demand_col = f'demand_{item_id}'
        profit_col = f'profit_{item_id}'
        
        # Validate required columns exist
        required_cols = [demand_col, profit_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for item {item_id}: {missing_cols}")
        
        # Get data
        demands = df[demand_col].tolist()
        profit = float(df[profit_col].iloc[0])
        
        total_periods = len(demands)
        total_demand = sum(demands)
        
        # Simple formula: Perfect Score = sum(demand) × profit
        perfect_score = profit * total_demand
        
        results[item_id] = {
            'total_periods': total_periods,
            'profit': profit,
            'total_demand': total_demand,
            'perfect_score': perfect_score
        }
        
        total_perfect_score += perfect_score
    
    results['_total'] = {
        'total_perfect_score': total_perfect_score
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Calculate perfect score for vending machine test case')
    parser.add_argument('--demand-file', type=str, required=True,
                       help='Path to CSV file with demand data (test.csv)')
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.demand_file):
        print(f"Error: File not found: {args.demand_file}")
        sys.exit(1)
    
    print("=" * 70)
    print("=== Perfect Score Calculator ===")
    print("=" * 70)
    print(f"\nInput file: {args.demand_file}")
    print(f"\nFormula: Perfect Score = sum(all demand) * profit")
    
    # Calculate perfect score
    results = calculate_perfect_score(args.demand_file)
    
    # Display results for each item
    for item_id, data in results.items():
        if item_id == '_total':
            continue
            
        print(f"\n{'-' * 70}")
        print(f"Item: {item_id}")
        print(f"{'-' * 70}")
        print(f"  Total Periods: {data['total_periods']}")
        print(f"  Total Demand: {data['total_demand']}")
        print(f"  Profit per unit: ${data['profit']}")
        print(f"  Perfect Score: {data['total_demand']} * ${data['profit']} = ${data['perfect_score']:.2f}")
    
    # Display total
    print("\n" + "=" * 70)
    print("=== TOTAL PERFECT SCORE ===")
    print("=" * 70)
    print(f"\n>>> Perfect Score: ${results['_total']['total_perfect_score']:.2f} <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()

