import os
import pandas as pd
from datetime import datetime, timedelta
from strategy_utils import (
    add_indicators, breakout_signal,
    bb_breakout_signal, bb_pullback_signal,
    combine_signals
)

DATA_DIR = "swing_data"
SYMBOLS = ["RELIANCE", "SUNPHARMA"]  # Extend as needed
INITIAL_CAPITAL = 1_000_000

def load_candle_data(symbol, years=2):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=['date'])
    cutoff = datetime.now() - timedelta(days=365*years)
    df = df[df['date'] >= cutoff].sort_values('date').reset_index(drop=True)
    return df

def run_full_backtest(symbol, capital=INITIAL_CAPITAL):
    df = load_candle_data(symbol)
    df = add_indicators(df)
    df = breakout_signal(df)
    df = bb_breakout_signal(df)
    df = bb_pullback_signal(df)
    df = combine_signals(df)

    # Backtesting logic that simulates trading on signals from df,
    # consuming capital, sizing positions, applying stops, exits, profit targets,
    # similar to your backtest function.

    # Example using your existing backtest implementation:
    trades = backtest(df, symbol)

    # You can extend backtest() to accept capital and position size params.

    return trades

def main():
    all_trades = []
    for symbol in SYMBOLS:
        print(f"\nBacktesting {symbol} for 2 years...")
        trades = run_full_backtest(symbol)
        if trades:
            all_trades.extend(trades)
    
    # Summarize performance
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        total_trades = len(trades_df)
        total_pnl = trades_df['pnl'].sum()
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        print(f"\nTotal trades: {total_trades}")
        print(f"Overall PnL: {total_pnl:.2f}")
        print(f"Win Rate: {win_rate:.2f}%")
        print("\nPnL by Entry Type:")
        print(trades_df.groupby('entry_type')['pnl'].agg(['count', 'sum', 'mean']))
        print("\nExit Reason Performance:")
        print(trades_df.groupby('exit_reason')['pnl'].agg(['count', 'sum', 'mean']))
    else:
        print("No trades executed in backtest")

if __name__ == "__main__":
    main()
