import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configurationimport pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATADIR = 'swing_data'
END_DATE = datetime(2025, 8, 31)
START_DATE = END_DATE - timedelta(days=5*365)

ema_configs = [
    {'atr_mult': 1.5, 'period': 10, 'short_ema': 10, 'long_ema': 21},
]

# Check if the data directory exists
if not os.path.exists(DATADIR):
    raise FileNotFoundError(f"Data directory not found: {DATADIR}")
all_files = [f for f in os.listdir(DATADIR) if f.endswith('.csv')]

# Load data for each symbol
def load_data_for_backtest(directory, symbol_file, start_date, end_date):
    filepath = os.path.join(directory, symbol_file)
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = df.sort_values('date').reset_index(drop=True)
    # Calculate EMAs
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    # Initialize rejection reason column
    df['rejection_reason'] = ''
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

# Backtest strategy with reason tracking
def backtest_strategy(df, symbol, config):
    df = supertrend(df, config['period'], config['atr_mult'])
    df['position'] = 0
    df['entry_price'] = np.nan
    df['profit'] = 0
    position = 0
    trail_stop = np.nan
    entry_price = np.nan

    for i in range(1, len(df)):
        # Initialize reasons
        reasons = []
        ema_cross = df['ema_10'].iloc[i] > df['ema_21'].iloc[i] and df['ema_10'].iloc[i-1] <= df['ema_21'].iloc[i-1]
        if not ema_cross:
            reasons.append('EMA cross failed')
        above_200d = df['close'].iloc[i] > df['ema_200'].iloc[i]
        if not above_200d:
            reasons.append('Below 200EMA')
        supertrend_signal = df['supertrend'].iloc[i] and not df['supertrend'].iloc[i-1]
        if not supertrend_signal:
            reasons.append('Supertrend not green')

        # If all conditions are met, enter trade
        if ema_cross and above_200d and supertrend_signal and position == 0:
            position = 1
            df.at[i, 'position'] = 1
            entry_price = df['close'].iloc[i]
            df.at[i, 'entry_price'] = entry_price
            trail_stop = df['lowerband'].iloc[i]
            df.at[i, 'profit'] = 0
        else:
            df.at[i, 'rejection_reason'] = ', '.join(reasons)

        # Exit logic (same as before)
        ema_cross_down = df['ema_10'].iloc[i] < df['ema_21'].iloc[i] and df['ema_10'].iloc[i-1] >= df['ema_21'].iloc[i-1]
        if position == 1 and (ema_cross_down or df['close'].iloc[i] <= trail_stop):
            df.at[i, 'position'] = 0
            position = 0
            df.at[i, 'profit'] = df['close'].iloc[i] - entry_price

        if position == 1 and df['lowerband'].iloc[i] > trail_stop:
            trail_stop = df['lowerband'].iloc[i]

    return df

# Backtest for all symbols, save detailed results for each
for file in all_files:
    symbol = file.split('.')[0]
    df = load_data_for_backtest(DATADIR, file, START_DATE, END_DATE)
    if df.empty:
        continue
    result = backtest_strategy(df.copy(), symbol, ema_configs[0])
    result.to_csv(f'rejection_reasons_{symbol}.csv', index=False)
    print(f"Saved rejection reasons for {symbol}")

print("Backtest with trade rejection reasons complete. Check CSV files for details.")

DATADIR = 'swing_data'  # Use the correct path
END_DATE = datetime(2025, 8, 31)
START_DATE = END_DATE - timedelta(days=5*365)

# Parameters for testing
ema_configs = [
    {'atr_mult': 1.5, 'period': 10, 'short_ema': 10, 'long_ema': 21},
    {'atr_mult': 2.0, 'period': 10, 'short_ema': 10, 'long_ema': 21},
    {'atr_mult': 2.5, 'period': 14, 'short_ema': 12, 'long_ema': 25},
]

# Check if the data directory exists
if not os.path.exists(DATADIR):
    raise FileNotFoundError(f"Data directory not found: {DATADIR}")
all_files = [f for f in os.listdir(DATADIR) if f.endswith('.csv')]

# Load data for each symbol
def load_data_for_backtest(directory, symbol_file, start_date, end_date):
    filepath = os.path.join(directory, symbol_file)
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = df.sort_values('date').reset_index(drop=True)
    # Calculate EMAs
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    return df

# Supertrend indicator
def supertrend(df, period=10, multiplier=3):
    df['tr'] = np.max((df['high'] - df['low'],
                      abs(df['high'] - df['close'].shift(1)),
                      abs(df['low'] - df['close'].shift(1))), axis=0)
    df['atr'] = df['tr'].ewm(span=period).mean()  # Use EMA for ATR
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

# Backtest strategy with exit on EMA crossover
def backtest_strategy(df, symbol, config):
    df = supertrend(df, config['period'], config['atr_mult'])
    df['ema_10'] = df['close'].ewm(span=config['short_ema']).mean()
    df['ema_21'] = df['close'].ewm(span=config['long_ema']).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    df['position'] = 0
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['returns'] = np.nan
    df['cumulative_returns'] = 1
    df['profit'] = 0
    position = 0
    trail_stop = np.nan
    entry_price = np.nan

    for i in range(1, len(df)):
        # Entry conditions: EMA cross up, above 200 EMA, Supertrend flipped
        ema_cross = df['ema_10'].iloc[i] > df['ema_21'].iloc[i] and df['ema_10'].iloc[i-1] <= df['ema_21'].iloc[i-1]
        above_200d = df['close'].iloc[i] > df['ema_200'].iloc[i]
        supertrend_signal = df['supertrend'].iloc[i] and not df['supertrend'].iloc[i-1]

        if ema_cross and above_200d and supertrend_signal and position == 0:
            position = 1
            df.at[i, 'position'] = 1
            entry_price = df['close'].iloc[i]
            df.at[i, 'entry_price'] = entry_price
            trail_stop = df['lowerband'].iloc[i]
            df.at[i, 'stop_loss'] = trail_stop
            df.at[i, 'profit'] = 0

        # Exit on EMA cross down (10 EMA below 21 EMA)
        ema_cross_down = df['ema_10'].iloc[i] < df['ema_21'].iloc[i] and df['ema_10'].iloc[i-1] >= df['ema_21'].iloc[i-1]
        if position == 1 and (ema_cross_down or df['close'].iloc[i] <= trail_stop):
            df.at[i, 'position'] = 0
            position = 0
            df.at[i, 'profit'] = df['close'].iloc[i] - entry_price

        if position == 1 and df['lowerband'].iloc[i] > trail_stop:
            trail_stop = df['lowerband'].iloc[i]
            df.at[i, 'stop_loss'] = trail_stop

        df.at[i, 'returns'] = df['profit'].iloc[i]
        df.at[i, 'cumulative_returns'] = df['cumulative_returns'].iloc[i-1] * (1 + df['profit'].iloc[i] / entry_price if entry_price != np.nan and df['profit'].iloc[i] != 0 else 1)

    df['symbol'] = symbol
    return df

# Backtest for all configs and symbols
all_configs = []
for config in ema_configs:
    combined_results = pd.DataFrame()
    for file in all_files:
        symbol = file.split('.')[0]
        df = load_data_for_backtest(DATADIR, file, START_DATE, END_DATE)
        if df.empty:
            continue
        result = backtest_strategy(df.copy(), symbol, config)
        combined_results = pd.concat([combined_results, result], ignore_index=True)
    
    if not combined_results.empty:
        total_trades = combined_results[combined_results['profit'] != 0]['profit'].count()
        winners = combined_results[combined_results['profit'] > 0]['profit']
        losers = combined_results[combined_results['profit'] < 0]['profit']
        total_profit = combined_results['profit'].sum()
        gross_profit = winners.sum() if not winners.empty else 0
        gross_loss = abs(losers.sum()) if not losers.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
        sharpe = combined_results['profit'].mean() / combined_results['profit'].std() if combined_results['profit'].std() > 0 else 0
        sortino = combined_results['profit'].mean() / combined_results[combined_results['profit'] < 0]['profit'].std() if not combined_results[combined_results['profit'] < 0].empty else 0

        all_configs.append({
            'atr_mult': config['atr_mult'],
            'ema_short': config['short_ema'],
            'ema_long': config['long_ema'],
            'total_trades': total_trades,
            'total_profit': total_profit,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': round(profit_factor, 2) if not np.isnan(profit_factor) else 0,
            'sharpe_ratio': round(sharpe, 2),
            'sortino_ratio': round(sortino, 2),
            'net_pnl': total_profit,
        })

# Export results
all_configs_df = pd.DataFrame(all_configs)
all_configs_df.to_csv('ema_exit_summary.csv', index=False)
print("Backtest with EMA exit complete.")
