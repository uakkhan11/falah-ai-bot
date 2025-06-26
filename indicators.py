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

def calculate_trailing_sl(prices, atr=1.5):
    """
    Calculate trailing stoploss based on highest price minus ATR multiplier.
    'prices' should be a list or Series of historical prices.
    """
    if len(prices) < 2:
        return None

    high_price = max(prices)
    low_price = min(prices)
    atr_value = (high_price - low_price) / len(prices)  # Simple ATR proxy
    trailing_sl = high_price - (atr * atr_value)
    return round(trailing_sl, 2)

def check_supertrend_flip(df):
    """
    Dummy Supertrend Flip Detector – Replace with real logic later.
    For now, returns True if last candle close is below open (i.e., red candle).
    """
    if df.empty or len(df) < 2:
        return False

    return df['close'].iloc[-1] < df['open'].iloc[-1]

