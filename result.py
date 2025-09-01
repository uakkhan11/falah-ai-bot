import pandas as pd

trades = pd.read_csv("walk_forward_trades.csv")
if trades.empty:
    raise SystemExit("No trades to summarize")

def extract_trade_stats(df):
    return {
        'Total Trades': len(df),
        'Winning Trades': int((df['pnl'] > 0).sum()),
        'Losing Trades': int((df['pnl'] <= 0).sum()),
        'Win Rate %': round((df['pnl'] > 0).mean() * 100, 2),
        'Avg PnL per Trade': round(df['pnl'].mean(), 2),
        'Best PnL': round(df['pnl'].max(), 2),
        'Worst PnL': round(df['pnl'].min(), 2),
        'Total PnL': round(df['pnl'].sum(), 2),
    }

stats = trades.groupby('symbol').apply(extract_trade_stats).apply(pd.Series).reset_index()
stats = stats.rename(columns={'symbol': 'Symbol'})
stats['Strategy'] = 'ML+All Features Walk-Forward'
stats.to_csv('walk_forward_stats.csv', index=False)
print(stats.head())
