def vwap_bounce_strategy(df):
    df = df.copy()
    df['Signal'] = (
        (df['close'] > df['VWAP']) &
        (df['close'].shift(1) < df['VWAP'].shift(1)) &  # price bouncing
        (df['RSI'] > 50)
    )
    return df

def rsi_breakout_strategy(df):
    df = df.copy()
    df['Signal'] = (
        (df['RSI'] > 60) &
        (df['RSI'].shift(1) <= 60) &
        (df['EMA10'] > df['EMA21'])
    )
    return df
