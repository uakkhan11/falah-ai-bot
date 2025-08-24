import os
import pandas as pd
from strategy_utils import (
    add_indicators, breakout_signal,
    bb_breakout_signal, bb_pullback_signal,
    combine_signals
)

# Adjust these paths to your data folders
BACKTEST_DIR = "backtest_signals"  # CSVs with saved backtest signals per symbol
LIVE_DATA_DIR = "swing_data"       # Your live data folder

def load_backtest_signals(symbol):
    path = os.path.join(BACKTEST_DIR, f"{symbol}_signals.csv")
    if not os.path.exists(path):
        print(f"Backtest signal file not found for {symbol}")
        return None
    return pd.read_csv(path, parse_dates=['date'])

def load_live_signals(symbol):
    path = os.path.join(LIVE_DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"Live data file not found for {symbol}")
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = add_indicators(df)
    df = breakout_signal(df)
    df = bb_breakout_signal(df)
    df = bb_pullback_signal(df)
    df = combine_signals(df)
    # Return only relevant columns to compare
    return df[['date', 'entry_signal', 'entry_type', 'breakout_signal', 'bb_breakout_signal', 'bb_pullback_signal']]

def compare_signals(symbol):
    bt = load_backtest_signals(symbol)
    live = load_live_signals(symbol)
    if bt is None or live is None:
        return
    # Merge on date
    merged = live.merge(bt, on='date', suffixes=('_live', '_bt'))
    if merged.empty:
        print(f"No overlapping dates for {symbol}")
        return

    # Compare columns
    cols_to_check = ['entry_signal', 'entry_type', 'breakout_signal', 'bb_breakout_signal', 'bb_pullback_signal']
    for col in cols_to_check:
        col_live = f"{col}_live"
        col_bt = f"{col}_bt"
        mismatches = merged[merged[col_live] != merged[col_bt]]
        if not mismatches.empty:
            print(f"\nMISMATCHES for symbol {symbol} on column {col}:")
            print(mismatches[['date', col_live, col_bt]])

def main():
    # List of symbols to validate
    symbols = ["RELIANCE", "TCS", "INFY"]  # Add more symbols as needed

    for symbol in symbols:
        print(f"\nValidating signals for {symbol}...")
        compare_signals(symbol)
    print("\nValidation complete.")

if __name__ == "__main__":
    main()
