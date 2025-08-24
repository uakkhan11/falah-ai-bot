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

    # Entry signal (Boolean 1/0)
    df['entry_signal'] = (
        (df['macd_line'] > 0) &
        (df['macd_signal'] > 0) &
        (df['rsi_14'] >= 40) & (df['rsi_14'] <= 70) &
        (df['close'] >= df['bb_lower'])
    ).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
