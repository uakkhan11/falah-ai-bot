import pandas as pd

def calculate_supertrend(df, period=10, multiplier=2):
    df['TR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(period).mean()
    hl2 = (df['High'] + df['Low']) / 2
    df['Upper Basic'] = hl2 + (multiplier * df['ATR'])
    df['Lower Basic'] = hl2 - (multiplier * df['ATR'])

    df['Upper Band'] = df['Upper Basic']
    df['Lower Band'] = df['Lower Basic']

    for i in range(1, len(df)):
        if df['Close'][i - 1] > df['Upper Band'][i - 1]:
            df['Upper Band'][i] = min(df['Upper Basic'][i], df['Upper Band'][i - 1])
        if df['Close'][i - 1] < df['Lower Band'][i - 1]:
            df['Lower Band'][i] = max(df['Lower Basic'][i], df['Lower Band'][i - 1])

    df['Supertrend'] = df['Upper Band']
    for i in range(1, len(df)):
        if df['Close'][i - 1] <= df['Supertrend'][i - 1]:
            df['Supertrend'][i] = df['Upper Band'][i]
        else:
            df['Supertrend'][i] = df['Lower Band'][i]

    df.drop(['TR', 'ATR', 'Upper Basic', 'Lower Basic', 'Upper Band', 'Lower Band'], axis=1, inplace=True)
    return df
