import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuration: Update path to match your directory structure
DATA_DIR = 'falah-ai-bot/swing_data'  # Change this to your actual path
END_DATE = datetime(2025, 8, 31)
START_DATE = END_DATE - timedelta(days=5*365)

# Load all CSV files in the specified directory
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

# Load data for each symbol
def load_data_for_backtest(directory, symbol_file, start_date, end_date):
    filepath = os.path.join(directory, symbol_file)
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = df.sort_values('date').reset_index(drop=True)
    return df

# Supertrend indicator
def supertrend(df, period=10, multiplier=1.5):
    df['tr'] = np.max((df['high'] - df['low'],
                      abs(df['high'] - df['close'].shift(1)),
                      abs(df['low'] - df['close'].shift(1))), axis=0)
    df['atr'] = df['tr'].rolling(period).mean()
    hl2 = (df['high'] + df['low']) / 2
    df['upperband'] = hl2 + (multiplier * df['atr'])
    df['lowerband'] = hl2 - (multiplier * df['atr'])
    df['supertrend'] = True
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i-1]:
            df['supertrend'].iloc[i] = True
        elif df['close'].iloc[i] < df['lowerband'].iloc[i-1]:
            df['supertrend'].iloc[i] = False
        else:
            df['supertrend'].iloc[i] = df['supertrend'].iloc[i-1]
    return df

# Backtest strategy with trailing stop loss and pyramiding
def backtest_strategy(df, symbol, period=10, multiplier=1.5):
    df = supertrend(df, period, multiplier)
    df['position'] = 0
    df['pyramid_count'] = 0
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['returns'] = np.nan
    df['cumulative_returns'] = 1
    df['profit'] = 0
    df['symbol'] = symbol
    position = 0
    pyramid_count = 0
    trail_stop = np.nan
    entry_price = np.nan

    for i in range(1, len(df)):
        if df['supertrend'].iloc[i] and not df['supertrend'].iloc[i-1] and position == 0:
            position = 1
            df.at[i, 'position'] = 1
            entry_price = df['close'].iloc[i]
            df.at[i, 'entry_price'] = entry_price
            trail_stop = df['lowerband'].iloc[i]
            df.at[i, 'stop_loss'] = trail_stop
            df.at[i, 'profit'] = 0
        elif position == 1 and df['close'].iloc[i] > df['close'].iloc[i-1]:
            pyramid_count += 1
            df.at[i, 'pyramid_count'] = pyramid_count
        if position == 1:
            if not df['supertrend'].iloc[i] or df['close'].iloc[i] <= trail_stop:
                df.at[i, 'position'] = 0
                position = 0
                df.at[i, 'profit'] = df['close'].iloc[i] - entry_price
                pyramid_count = 0
            elif df['lowerband'].iloc[i] > trail_stop:
                trail_stop = df['lowerband'].iloc[i]
                df.at[i, 'stop_loss'] = trail_stop
        else:
            df.at[i, 'profit'] = 0

        df.at[i, 'returns'] = df['profit'].iloc[i]
        df.at[i, 'cumulative_returns'] = df['cumulative_returns'].iloc[i-1] * (1 + df['profit'].iloc[i] / entry_price if entry_price != np.nan and df['profit'].iloc[i] != 0 else 1)

    return df

# Backtest all symbols in folder
combined_results = pd.DataFrame()
summary_results = []

for file in all_files:
    symbol = file.split('.')[0]
    df = load_data_for_backtest(DATADIR, file, START_DATE, END_DATE)
    if df.empty:
        continue
    result = backtest_strategy(df.copy(), symbol, period=10, multiplier=1.5)
    combined_results = pd.concat([combined_results, result], ignore_index=True)

    # Summary for each symbol
    total_profit = result['profit'].sum()
    wins = (result['profit'] > 0).sum()
    losses = (result['profit'] < 0).sum()
    trade_count = result['profit'].replace(0, np.nan).dropna().count()
    win_rate = 0 if trade_count == 0 else (wins / trade_count) * 100
    summary_results.append({
        "Symbol": symbol,
        "Total Trades": trade_count,
        "Total Profit": total_profit,
        "Winning Trades": wins,
        "Losing Trades": losses,
        "Win Rate": f"{win_rate:.2f}%",
        "Cumulative Return": f"{result['cumulative_returns'].iloc[-1] if not result.empty else 1:.2f}"
    })

# Save results
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv('all_symbols_summary.csv', index=False)
combined_results.to_csv('all_symbols_backtest_detailed.csv', index=False)

print("Backtest complete. Summary and detailed results saved.")
