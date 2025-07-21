import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
import pandas_ta as pta


def calculate_rsi(df, period=14):
    return RSIIndicator(close=df["close"], window=period).rsi()


def calculate_ema(df, period):
    return EMAIndicator(close=df["close"], window=period).ema_indicator()


def calculate_macd(df):
    macd = MACD(close=df["close"])
    return macd.macd(), macd.macd_signal()


def detect_macd_bullish_cross(df):
    macd, signal = calculate_macd(df)
    return macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]


def calculate_atr(df, period=14):
    return AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=period).average_true_range()


def calculate_supertrend(df, period=10, multiplier=3):
    st = pta.supertrend(df["high"], df["low"], df["close"], length=period, multiplier=multiplier)
    trend_col = [col for col in st.columns if 'SUPERTd' in col][0]
    return st[trend_col]


def detect_supertrend_green(df):
    st_dir = calculate_supertrend(df)
    return st_dir.iloc[-1] == 1


def detect_breakout(df, threshold=1.02):
    if len(df) < 2:
        return False, None
    prev_high = df['high'].iloc[-2]
    curr_close = df['close'].iloc[-1]
    is_breakout = curr_close > prev_high * threshold
    return is_breakout, round(prev_high * threshold, 2)


def detect_3green_days(df):
    if len(df) < 3:
        return False
    last_3 = df.iloc[-3:]
    return all(last_3['close'] > last_3['open'])


def detect_darvas_box(df, lookback=20):
    if len(df) < lookback + 1:
        return False, None
    recent_high = df['high'].iloc[-lookback:-1].max()
    curr_close = df['close'].iloc[-1]
    is_breakout = curr_close > recent_high
    return is_breakout, round(recent_high, 2)


def detect_bullish_pivot(df, lookback=5):
    """
    Detects bullish pivot where recent low is a higher low compared to previous pivot low.
    """
    if len(df) < lookback + 2:
        return False

    pivot_lows = []
    for i in range(lookback, len(df) - 1):
        if df['low'].iloc[i] < df['low'].iloc[i - 1] and df['low'].iloc[i] < df['low'].iloc[i + 1]:
            pivot_lows.append((i, df['low'].iloc[i]))

    if len(pivot_lows) < 2:
        return False

    last_pivot = pivot_lows[-1][1]
    prev_pivot = pivot_lows[-2][1]

    return last_pivot > prev_pivot


def detect_recent_high(df, lookback=20):
    """
    Detect highest close in recent period (used to trail profits near highs)
    """
    if len(df) < lookback:
        return None
    return df["close"].iloc[-lookback:].max()


def detect_recent_swing_low(df, lookback=5):
    """
    Detect lowest low in recent candles (protective trailing SL)
    """
    if len(df) < lookback:
        return None
    return df["low"].iloc[-lookback:].min()


def calculate_chandelier_exit(df, atr_multiplier=3):
    """
    Chandelier Exit = Highest Close - ATR * multiplier (for exit trailing)
    """
    atr = calculate_atr(df)
    highest_close = df["close"].rolling(window=22).max()
    ce = highest_close - atr * atr_multiplier
    return ce
