import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Simulate sample data for backtesting
np.random.seed(42)
rows = 500
price = 100
prices = [price]
for i in range(1, rows):
    price = price + np.random.normal(0.2, 1.5)
    prices.append(price)
df = pd.DataFrame({'close': prices})
df['open'] = df['close'].shift(1) + np.random.uniform(-0.5, 0.5, len(df))
df['high'] = np.maximum(df['open'], df['close']) + np.random.uniform(0, 0.5, len(df))
df['low'] = np.minimum(df['open'], df['close']) - np.random.uniform(0, 0.5, len(df))

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

def backtest_strategy(df, period=10, multiplier=1.5):
    df = supertrend(df, period, multiplier)
    df['position'] = 0
    df['pyramid_count'] = 0
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['returns'] = np.nan
    df['cumulative_returns'] = 1
    df['profit'] = 0
    position = 0
    pyramid_count = 0
    trail_stop = np.nan
    entry_price = np.nan

    for i in range(1, len(df)):
        # Entry: Supertrend turns green
        if df['supertrend'].iloc[i] and not df['supertrend'].iloc[i-1] and position == 0:
            position = 1
            df.at[i, 'position'] = 1
            entry_price = df['close'].iloc[i]
            df.at[i, 'entry_price'] = entry_price
            trail_stop = df['lowerband'].iloc[i]
            df.at[i, 'stop_loss'] = trail_stop
            df.at[i, 'profit'] = 0

        # Pyramiding: Count consecutive up closes
        elif position == 1 and df['close'].iloc[i] > df['close'].iloc[i-1]:
            pyramid_count += 1
            df.at[i, 'pyramid_count'] = pyramid_count

        # Exit: Supertrend turns red or trailing stop hit
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

# Run backtest
result = backtest_strategy(df.copy(), period=10, multiplier=1.5)

# Calculate summary statistics
summary = {
    "Total Trades": result['profit'].replace(0, np.nan).dropna().count(),
    "Total Profit": result['profit'].sum(),
    "Winning Trades": (result['profit'] > 0).sum(),
    "Losing Trades": (result['profit'] < 0).sum(),
    "Win Rate": f"{(result['profit'] > 0).sum() / result['profit'].replace(0, np.nan).dropna().count() * 100:.2f}%" if result['profit'].replace(0, np.nan).dropna().count() > 0 else "0%",
    "Max Profit per Trade": result['profit'].max(),
    "Max Loss per Trade": result['profit'].min(),
    "Average Profit per Trade": result['profit'].mean(),
    "Cumulative Return": f"{result['cumulative_returns'].iloc[-1]:.2f}"
}

# Export results
summary_df = pd.DataFrame([summary])
summary_df.to_csv('backtest_summary.csv', index=False)
result.to_csv('backtest_detailed.csv', index=False)

# Display summary
print(summary_df)
