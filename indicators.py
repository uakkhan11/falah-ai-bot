import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from ta.trend import PSARIndicator
from ta.trend import SuperTrend

def detect_breakout(df, threshold=1.02):
    """
    Detects breakout if current close > previous high * threshold.
    Returns bool and breakout level.
    """
    if len(df) < 2:
        return False, None
    prev_high = df['High'].iloc[-2]
    curr_close = df['Close'].iloc[-1]
    is_breakout = curr_close > prev_high * threshold
    return is_breakout, round(prev_high * threshold, 2)

def detect_rsi_ema_signals(df, rsi_period=14, ema_period=21, rsi_threshold=60):
    """
    Returns True if RSI > threshold and Close > EMA.
    """
    rsi = RSIIndicator(df['Close'], window=rsi_period).rsi()
    ema = EMAIndicator(df['Close'], window=ema_period).ema_indicator()
    last_rsi = rsi.iloc[-1]
    last_close = df['Close'].iloc[-1]
    last_ema = ema.iloc[-1]
    return last_rsi > rsi_threshold and last_close > last_ema

def detect_3green_days(df):
    """
    Detects 3 consecutive green candles.
    """
    if len(df) < 3:
        return False
    last_3 = df.iloc[-3:]
    return all(last_3['Close'] > last_3['Open'])

def detect_darvas_box(df, lookback=20):
    """
    Detects if close breaks above the high of the last 'lookback' candles.
    Returns bool and breakout level.
    """
    if len(df) < lookback + 1:
        return False, None
    recent_high = df['High'].iloc[-lookback:-1].max()
    curr_close = df['Close'].iloc[-1]
    is_breakout = curr_close > recent_high
    return is_breakout, round(recent_high, 2)

def calculate_trailing_sl(prices, atr_multiplier=1.5):
    """
    Calculate trailing stoploss based on highest price minus ATR.
    """
    if len(prices) < 2:
        return None
    high_price = max(prices)
    low_price = min(prices)
    atr_value = (high_price - low_price) / len(prices)
    trailing_sl = high_price - (atr_multiplier * atr_value)
    return round(trailing_sl, 2)

def check_supertrend_flip(df, period=10, multiplier=3):
    """
    Checks if Supertrend has flipped to bearish.
    Returns True if last candle is bearish.
    """
    if len(df) < period + 1:
        return False
    st = SuperTrend(df['High'], df['Low'], df['Close'], period=period, multiplier=multiplier)
    supertrend = st.super_trend()
    direction = st.super_trend_direction()
    # -1 means bearish, 1 means bullish
    last_dir = direction.iloc[-1]
    return last_dir == -1

def detect_macd_cross(df):
    """
    Detects MACD bullish cross (MACD line crosses above Signal line).
    """
    if len(df) < 35:
        return False
    macd = MACD(df['Close'])
    macd_line = macd.macd()
    signal_line = macd.macd_signal()
    if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        return True
    return False
