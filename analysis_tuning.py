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
for g in gate_cols:
    combo = filtered[filtered[g]]
    num_trades = len(combo)
    if num_trades > 0:
        win_rate = combo['profitable'].mean()
        pf = combo[combo['pnl'] > 0]['pnl'].sum() / abs(combo[combo['pnl'] < 0]['pnl'].sum()) if len(combo[combo['pnl'] < 0]) > 0 else float('nan')
        print(f"{g}: trades={num_trades}, win_rate={win_rate:.2%}, profit_factor={pf:.2f}")
combo_df = pd.DataFrame(best_results)
if combo_df.empty:
    print("No qualifying gate combos found (try reducing the minimum trade threshold).")
else:
    combo_df = combo_df.sort_values(by='profit_factor', ascending=False)
    print(combo_df.head(10))
