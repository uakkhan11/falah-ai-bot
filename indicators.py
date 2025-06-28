# indicators.py
import pandas as pd

def detect_breakout(df, threshold=1.02):
    """
    Detects breakout if current close > previous high * threshold.
    """
    if len(df) < 2:
        return False
    prev_high = df['High'].iloc[-2]
    curr_close = df['Close'].iloc[-1]
    return curr_close > prev_high * threshold

def detect_rsi_ema_signals(df, rsi_period=14, ema_period=21):
    """
    Returns True if RSI > 60 and Close > EMA.
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    ema = df['Close'].ewm(span=ema_period, adjust=False).mean()
    last_rsi = rsi.iloc[-1]
    last_close = df['Close'].iloc[-1]
    last_ema = ema.iloc[-1]
    return last_rsi > 60 and last_close > last_ema

def detect_3green_days(df):
    """
    Detects 3 consecutive green candles.
    """
    if len(df) < 3:
        return False
    last_3 = df.iloc[-3:]
    return all(row['Close'] > row['Open'] for idx, row in last_3.iterrows())

def detect_darvas_box(df, lookback=20):
    """
    Detects if close breaks above the high of the last 'lookback' candles.
    """
    if len(df) < lookback + 1:
        return False
    recent_high = df['High'].iloc[-lookback:-1].max()
    curr_close = df['Close'].iloc[-1]
    return curr_close > recent_high

def calculate_trailing_sl(prices, atr=1.5):
    """
    Calculate trailing stoploss based on highest price minus ATR multiplier.
    """
    if len(prices) < 2:
        return None
    high_price = max(prices)
    low_price = min(prices)
    atr_value = (high_price - low_price) / len(prices)
    trailing_sl = high_price - (atr * atr_value)
    return round(trailing_sl, 2)

def check_supertrend_flip(df):
    """
    Dummy: returns True if last candle red.
    """
    return df['Close'].iloc[-1] < df['Open'].iloc[-1]
