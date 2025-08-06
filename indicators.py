# indicators.py

import pandas as pd
import numpy as np
import pandas_ta as ta
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

def calculate_bollinger_bands(close_series, window=20, num_std_dev=2):
    """Returns upper, middle, lower Bollinger Bands"""
    middle_band = close_series.rolling(window).mean()
    std = close_series.rolling(window).std()
    upper_band = middle_band + (num_std_dev * std)
    lower_band = middle_band - (num_std_dev * std)
    return upper_band, middle_band, lower_band

# --- Composite Indicator Additions (full set from both versions) ---

def add_all_indicators(df):
    # Pandas TA core indicators
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["EMA10"] = ta.ema(df["close"], length=10)
    df["EMA21"] = ta.ema(df["close"], length=21)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["MACD"] = ta.macd(df["close"])["MACDh_12_26_9"]
    df["Stochastic"] = ta.stoch(df["high"], df["low"], df["close"])["STOCHK_14_3"]
    df["VolumeChange"] = df["volume"].pct_change().fillna(0)

    # TA library extras
    df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

    # Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(df['close'], window=20, num_std_dev=2)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    return df

# --- Strategy Conditions (Relaxed Versions) ---

def detect_bullish_pivot(df, window=3):
    if len(df) < window + 2:
        return False
    recent = df.tail(window + 2)
    lows = recent["low"].values
    closes = recent["close"].values
    opens = recent["open"].values

    try:
        if (
            lows[1] < lows[0] and
            lows[1] < lows[2] and
            closes[2] > opens[2]
        ):
            return True
    except:
        return False
    return False

def detect_macd_bullish_cross(df):
    if "MACD" in df.columns and "macd_signal" in df.columns:
        macd = df["MACD"].tail(5).values
        signal = df["macd_signal"].tail(5).values
        for i in range(1, len(macd)):
            if macd[i - 1] < signal[i - 1] and macd[i] > signal[i]:
                return True
        return False

    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    for i in range(-4, -1):
        if macd.iloc[i - 1] < signal.iloc[i - 1] and macd.iloc[i] > signal.iloc[i]:
            return True
    return False

def detect_supertrend_green(df, lookback=3, period=10, multiplier=3):
    if "supertrend_direction" in df.columns:
        return (df["supertrend_direction"].tail(lookback) == "green").any()

    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range()
    hl2 = (df["high"] + df["low"]) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    close = df["close"]
    direction = []
    in_uptrend = True
    for i in range(len(close)):
        if i == 0:
            direction.append("green" if in_uptrend else "red")
            continue
        if close.iloc[i] > upperband.iloc[i - 1]:
            in_uptrend = True
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            in_uptrend = False
        direction.append("green" if in_uptrend else "red")
    return direction[-lookback:].count("green") > 0

# --- Intraday Signal Utilities ---

def intraday_vwap_bounce_strategy(df):
    df = df.copy()
    df['VWAP'] = (df['high'] + df['low'] + df['close']) / 3
    df['AboveVWAP'] = df['close'] > df['VWAP']
    signal = False

    if len(df) < 10:
        return df

    if (
        df['close'].iloc[-1] > df['VWAP'].iloc[-1] and
        df['low'].iloc[-2] < df['VWAP'].iloc[-2] and
        df['close'].iloc[-2] < df['VWAP'].iloc[-2]
    ):
        signal = True

    df['Signal'] = signal
    return df

def intraday_rsi_breakout_strategy(df, rsi_window=14, rsi_threshold=60):
    df = df.copy()
    df['RSI'] = RSIIndicator(close=df['close'], window=rsi_window).rsi()
    signal = False

    if df['RSI'].iloc[-2] < rsi_threshold and df['RSI'].iloc[-1] > rsi_threshold:
        signal = True

    df['Signal'] = signal
    return df
