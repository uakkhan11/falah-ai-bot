# indicators.py

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange

def add_indicators(df):
    """
    Adds basic indicators: RSI, EMA10, EMA21, ATR14, MACD to dataframe
    """
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    df["ema10"] = EMAIndicator(df["close"], window=10).ema_indicator()
    df["ema21"] = EMAIndicator(df["close"], window=21).ema_indicator()
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["atr14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    return df

def detect_macd_cross(df):
    if len(df) < 35:
        return False
    if df["macd"].iloc[-2] < df["macd_signal"].iloc[-2] and df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]:
        return True
    return False

def detect_rsi_signal(df, threshold=60):
    if len(df) < 15:
        return False
    return df["rsi"].iloc[-1] > threshold

def detect_ema_cross(df):
    if len(df) < 22:
        return False
    return df["close"].iloc[-1] > df["ema10"].iloc[-1] > df["ema21"].iloc[-1]

def detect_3_green_candles(df):
    if len(df) < 3:
        return False
    last3 = df.iloc[-3:]
    return all(last3["close"] > last3["open"])

def detect_breakout(df, threshold=1.02):
    if len(df) < 2:
        return False
    prev_high = df["high"].iloc[-2]
    curr_close = df["close"].iloc[-1]
    return curr_close > prev_high * threshold

def detect_darvas_breakout(df, lookback=20):
    if len(df) < lookback + 1:
        return False
    recent_high = df["high"].iloc[-lookback:-1].max()
    curr_close = df["close"].iloc[-1]
    return curr_close > recent_high
