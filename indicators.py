# indicators.py

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# --- Basic Calculators (used in filter logs/debug) ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# --- Composite Indicator Additions (used in batch) ---

def add_all_indicators(df):
    df['EMA10'] = EMAIndicator(close=df['close'], window=10).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    return df

# --- Strategy Conditions ---

def detect_bullish_pivot(df):
    """Check for bullish pivot - higher low and breakout candle."""
    try:
        recent = df.tail(5)
        if recent["close"].iloc[-1] > recent["high"].max():
            return True
    except Exception:
        pass
    return False

def detect_macd_bullish_cross(df):
    """MACD bullish crossover"""
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]

def detect_supertrend_green(df, period=10, multiplier=3):
    """Basic supertrend logic"""
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range()
    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    close = df["close"]
    trend = []
    in_uptrend = True
    for i in range(len(close)):
        if i == 0:
            trend.append(in_uptrend)
            continue
        if close.iloc[i] > upperband.iloc[i - 1]:
            in_uptrend = True
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            in_uptrend = False
        trend.append(in_uptrend)
    return trend[-1]

# --- Intraday Signal Utilities ---

def intraday_vwap_bounce_strategy(df):
    """Detect VWAP bounce pattern in intraday (15/60min) data"""
    df = df.copy()
    df['VWAP'] = (df['high'] + df['low'] + df['close']) / 3
    df['AboveVWAP'] = df['close'] > df['VWAP']
    signal = False

    if len(df) < 10:
        return df

    if (
        df['close'].iloc[-1] > df['VWAP'].iloc[-1] and  # above VWAP
        df['low'].iloc[-2] < df['VWAP'].iloc[-2] and    # tested VWAP prev candle
        df['close'].iloc[-2] < df['VWAP'].iloc[-2]      # closed below VWAP then bounced
    ):
        signal = True

    df['Signal'] = signal
    return df

def intraday_rsi_breakout_strategy(df, rsi_window=14, rsi_threshold=60):
    """RSI breakout above threshold on 15m/60m"""
    df = df.copy()
    df['RSI'] = RSIIndicator(close=df['close'], window=rsi_window).rsi()
    signal = False

    if df['RSI'].iloc[-2] < rsi_threshold and df['RSI'].iloc[-1] > rsi_threshold:
        signal = True

    df['Signal'] = signal
    return df
