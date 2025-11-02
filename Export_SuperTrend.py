import pandas as pd
import numpy as np

# Define SuperTrend function with loc assignment
def supertrend(df, period=10, multiplier=3):
    df['tr'] = np.max((df['high'] - df['low'],
                      abs(df['high'] - df['close'].shift(1)),
                      abs(df['low'] - df['close'].shift(1))), axis=0)
    df['atr'] = df['tr'].ewm(span=period).mean()
    hl2 = (df['high'] + df['low']) / 2
    df['upperband'] = hl2 + (multiplier * df['atr'])
    df['lowerband'] = hl2 - (multiplier * df['atr'])
    df['supertrend'] = 1  # Initialize as bullish
    for i in range(1, len(df)):
        if df.loc[i, 'close'] > df.loc[i-1, 'upperband']:
            df.loc[i, 'supertrend'] = 1  # Bullish
        elif df.loc[i, 'close'] < df.loc[i-1, 'lowerband']:
            df.loc[i, 'supertrend'] = -1  # Bearish
        else:
            df.loc[i, 'supertrend'] = df.loc[i-1, 'supertrend']
    return df

# Set your symbol
symbol = 'BHARTIARTL'  # Change to your symbol

# Load data
df = pd.read_csv(f'swing_data/{symbol}.csv')
df['date'] = pd.to_datetime(df['date'])

# Apply SuperTrend
df = supertrend(df, period=10, multiplier=3)

# Select required columns
df = df[['date', 'close', 'upperband', 'lowerband', 'supertrend']]

# Export to CSV
df.to_csv(f'{symbol}_supertrend_export.csv', index=False)
