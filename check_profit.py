import pandas as pd
from pathlib import Path
from collections import Counter

# Check profit distribution
all_profits = []
for f in Path('benchmark/real_trajectory/lead_time_0').glob('*/test.csv'):
    df = pd.read_csv(f)
    profit_col = [c for c in df.columns if 'profit' in c][0]
    all_profits.append(df[profit_col].iloc[0])

print('Profit distribution in lead_time_0:')
for profit, count in sorted(Counter(all_profits).items()):
    print(f'  profit={profit}: {count} instances')
