import pandas as pd

df = pd.read_csv("results_live_like/trades.csv")
df['profitable'] = (df['pnl'] > 0)
gate_cols = [c for c in df.columns if c.startswith("gate_") and df[c].dtype == bool]

results = []
for n in range(2, len(gate_cols) + 1):
    df['gate_sum'] = df[gate_cols].sum(axis=1)
    filt = df['gate_sum'] >= n
    trades = df[filt]
    if len(trades) == 0:
        continue
    win_rate = trades['profitable'].mean()
    profit_factor = trades[trades['pnl'] > 0]['pnl'].sum() / abs(trades[trades['pnl'] < 0]['pnl'].sum())
    results.append({'n_gates': n, 'num_trades': len(trades), 'win_rate': win_rate, 'profit_factor': profit_factor})

# Summary DataFrame
tune_df = pd.DataFrame(results)
print(tune_df)

symbol_stats = df.groupby('symbol')['profitable'].mean()
# Set your exclusion threshold (e.g., win rate < 0.18)
bad_symbols = symbol_stats[symbol_stats < 0.18].index.tolist()
print("Symbols to exclude:", bad_symbols)

filtered = df[~df['symbol'].isin(bad_symbols)]
print("Filtered win rate:", filtered['profitable'].mean())
print("Filtered profit factor:", filtered[filtered['pnl'] > 0]['pnl'].sum() / abs(filtered[filtered['pnl'] < 0]['pnl'].sum()))


from itertools import combinations

best_results = []
for g1, g2 in combinations(gate_cols, 2):
    combo = df[(df[g1]) & (df[g2])]
    if len(combo) < 10:
        continue  # skip tiny samples
    win_rate = combo['profitable'].mean()
    profit_factor = combo[combo['pnl'] > 0]['pnl'].sum() / abs(combo[combo['pnl'] < 0]['pnl'].sum())
    best_results.append({
    'gates': f"{g1}+{g2}",
    'num_trades': len(combo),
    'win_rate': win_rate,
    'profit_factor': profit_factor
})
combo_df = pd.DataFrame(best_results)
if combo_df.empty:
    print("No qualifying gate combos found (try reducing the minimum trade threshold).")
else:
    combo_df = combo_df.sort_values(by='profit_factor', ascending=False)
    print(combo_df.head(10))
