import os
import pandas as pd
from datetime import datetime, timedelta
from strategy_utils import (
    add_indicators, breakout_signal,
    bb_breakout_signal, bb_pullback_signal,
    combine_signals
)

# Data folder path
DATA_DIR = "swing_data"
BACKTEST_SIGNAL_DIR = "backtest_signals"

SYMBOLS = ["RELIANCE", "SUNPHARMA"]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(symbol, years=2):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(file_path):
        print(f"Warning: Data file not found for {symbol}: {file_path}")
        return None
    df = pd.read_csv(file_path, parse_dates=['date'])
    cutoff = datetime.now() - timedelta(days=365*years)
    df = df[df['date'] >= cutoff].sort_values('date').reset_index(drop=True)
    return df

def save_backtest_signals(symbol, df):
    ensure_dir(BACKTEST_SIGNAL_DIR)
    path = os.path.join(BACKTEST_SIGNAL_DIR, f"{symbol}_signals.csv")
    df.to_csv(path, index=False)
    print(f"Saved backtest signals: {path}")

def generate_signals_and_backtest(symbol):
    df = load_data(symbol)
    if df is None or df.empty:
        print(f"Skipping {symbol} due to no data")
        return None, None
    df = add_indicators(df)
    df = breakout_signal(df)
    df = bb_breakout_signal(df)
    df = bb_pullback_signal(df)
    df = combine_signals(df)
    # Save these signals for future comparison
    save_backtest_signals(symbol, df)
    # Run backtest on these signals (optional)
    trades = backtest(df, symbol)
    return df, trades

def compare_signals(live_df, backtest_df):
    cols = ['date', 'entry_signal', 'entry_type', 'breakout_signal', 'bb_breakout_signal', 'bb_pullback_signal']
    merged = live_df[cols].merge(backtest_df[cols], on='date', suffixes=('_live', '_bt'))
    mismatches = {}
    for col in cols[1:]:
        live_col = f"{col}_live"
        bt_col = f"{col}_bt"
        diff = merged[merged[live_col] != merged[bt_col]]
        if not diff.empty:
            mismatches[col] = diff[['date', live_col, bt_col]]
    return mismatches

def main():
    all_trades = []
    for symbol in SYMBOLS:
        print(f"\nProcessing {symbol}...")
        live_df, trades = generate_signals_and_backtest(symbol)
        if live_df is None:
            continue

        # Assume live_df is the "live" signals and we use same as backtest for demo
        backtest_df = pd.read_csv(os.path.join(BACKTEST_SIGNAL_DIR, f"{symbol}_signals.csv"), parse_dates=['date'])

        mismatches = compare_signals(live_df, backtest_df)

        if mismatches:
            print(f"Mismatches for {symbol}:")
            for col, diff in mismatches.items():
                print(f"\nColumn: {col}")
                print(diff)
        else:
            print(f"Signals match perfectly for {symbol}")

        if trades:
            all_trades.extend(trades)

    print(f"\nTotal trades from backtests: {len(all_trades)}")

if __name__ == "__main__":
    main()
