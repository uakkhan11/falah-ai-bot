import pandas_ta as ta

def add_indicators(df):
    df = df.copy()

    # MACD
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_line'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']

    # RSI
    df['rsi_14'] = ta.rsi(df['close'], length=14)

    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_lower'] = bbands['BBL_20_2.0']
    df['bb_upper'] = bbands['BBU_20_2.0']

    # ATR
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Supertrend
    supertrend = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
    df['supertrend_direction'] = supertrend['SUPERTd_10_3.0']

    # Chandelier Exit
    ch_length = 22
    atr_mult = 2.0
    highest_high = df['high'].rolling(ch_length).max()
    df['chandelier_exit'] = highest_high - atr_mult * df['atr']

    # Entry Signal: MACD positive, RSI 40-70, close above BB lower band
    df['entry_signal'] = (
        (df['macd_line'] > 0) &
        (df['macd_signal'] > 0) &
        (df['rsi_14'] >= 40) & (df['rsi_14'] <= 70) &
        (df['close'] >= df['bb_lower'])
    ).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def bullish_entry_filter(row):
    return (
        row['macd_line'] > 0 and
        row['macd_signal'] > 0 and
        40 <= row['rsi_14'] <= 70 and
        row['close'] >= row['bb_lower']
    )

def momentum_trend_exit(row):
    rsi_exit = (row['rsi_14'] < 70) and (row['close'] < row['bb_upper'])
    trend_exit = row['supertrend_direction'] < 0
    return rsi_exit or trend_exit
