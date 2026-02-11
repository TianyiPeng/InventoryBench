import pandas as pd
from pathlib import Path
from collections import Counter

batches = [
    'real_trajectory/lead_time_0',
    'real_trajectory/lead_time_4', 
    'real_trajectory/lead_time_stochastic',
    'synthetic_trajectory/lead_time_0',
    'synthetic_trajectory/lead_time_4',
    'synthetic_trajectory/lead_time_stochastic'
]

print("=" * 70)
print("Profit Distribution Across All Batches")
print("=" * 70)

for batch in batches:
    batch_path = Path(f'benchmark/{batch}')
    if not batch_path.exists():
        print(f"\n{batch}: NOT FOUND")
        continue
    
    all_profits = []
    # Use **/test.csv to handle nested directory structure in synthetic_trajectory
    for f in batch_path.glob('**/test.csv'):
        df = pd.read_csv(f)
        profit_col = [c for c in df.columns if 'profit' in c][0]
        all_profits.append(df[profit_col].iloc[0])
    
    counts = Counter(all_profits)
    print(f"\n{batch} ({len(all_profits)} instances):")
    for profit in sorted(counts.keys()):
        print(f"  profit={profit}: {counts[profit]} instances")
    
    # Check for unexpected values
    expected = {1, 4, 19}
    actual = set(counts.keys())
    if actual != expected:
        print(f"  ⚠️  WARNING: Expected {expected}, got {actual}")
