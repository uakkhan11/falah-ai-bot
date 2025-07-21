# indicators.py

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
import pandas_ta as pta

def detect_breakout(df, threshold=1.02):
    if len(df) < 2:
        return False, None
    prev_high = df['high'].iloc[-2]
    curr_close = df['close'].iloc[-1]
    is_breakout = curr_close > prev_high * threshold
    return is_breakout, round(prev_high * threshold, 2)

def detect_rsi_ema_signals(df, rsi_period=14, ema_period=21, rsi_threshold=60):
    rsi = RSIIndicator(df['close'], window=rsi_period).rsi()
    ema = EMAIndicator(df['close'], window=ema_period).ema_indicator()
    last_rsi = rsi.iloc[-1]
    last_close = df['close'].iloc[-1]
    last_ema = ema.iloc[-1]
    return last_rsi > rsi_threshold and last_close > last_ema

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

def calculate_trailing_sl(prices, atr_multiplier=1.5):
    if len(prices) < 2:
        return None
    high_price = max(prices)
    low_price = min(prices)
    atr_value = (high_price - low_price) / len(prices)
    trailing_sl = high_price - (atr_multiplier * atr_value)
    return round(trailing_sl, 2)

def check_supertrend_flip(kite, symbol, period=10, multiplier=3):
    try:
        instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
        hist = kite.historical_data(
            instrument_token=instrument_token,
            from_date=pd.Timestamp.today() - pd.Timedelta(days=30),
            to_date=pd.Timestamp.today(),
            interval="day"
        )
        df = pd.DataFrame(hist)
        supertrend = pta.supertrend(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=period,
            multiplier=multiplier
        )
        trend_col = [c for c in supertrend.columns if "SUPERT" in c and "d" in c][0]
        last_trend = supertrend[trend_col].iloc[-1]
        return last_trend == -1
    except Exception as e:
        print(f"⚠️ Supertrend check failed for {symbol}: {e}")
        return False

def detect_macd_cross(df):
    if len(df) < 35:
        return False
    macd = MACD(df['close'])
    macd_line = macd.macd()
    signal_line = macd.macd_signal()
    if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        return True
    return False

def calculate_atr_trailing_sl(kite, symbol, cmp, atr_multiplier=1.5):
    try:
        instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
        hist = kite.historical_data(
            instrument_token=instrument_token,
            from_date=pd.Timestamp.today() - pd.Timedelta(days=30),
            to_date=pd.Timestamp.today(),
            interval="day"
        )
        df = pd.DataFrame(hist)
        atr = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14
        ).average_true_range().iloc[-1]
        trailing_sl = round(cmp - atr * atr_multiplier, 2)
        return trailing_sl
    except Exception as e:
        print(f"⚠️ ATR trailing SL error for {symbol}: {e}")
        return None

def check_rsi_bearish_divergence(kite, symbol, lookback=5):
    try:
        instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
        hist = kite.historical_data(
            instrument_token=instrument_token,
            from_date=pd.Timestamp.today() - pd.Timedelta(days=30),
            to_date=pd.Timestamp.today(),
            interval="day"
        )
        df = pd.DataFrame(hist)
        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()

        if len(df) < lookback + 1:
            return False

        price_highs = df["close"].iloc[-lookback:]
        rsi_values = df["rsi"].iloc[-lookback:]

        if price_highs.is_monotonic_increasing and rsi_values.is_monotonic_decreasing:
            return True

        return False
    except Exception as e:
        print(f"⚠️ RSI divergence check failed for {symbol}: {e}")
        return False

def check_vwap_cross(kite, symbol):
    try:
        instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
        hist = kite.historical_data(
            instrument_token=instrument_token,
            from_date=pd.Timestamp.today() - pd.Timedelta(days=15),
            to_date=pd.Timestamp.today(),
            interval="day"
        )
        df = pd.DataFrame(hist)
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["cum_tp_vol"] = (df["typical_price"] * df["volume"]).cumsum()
        df["cum_vol"] = df["volume"].cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]
        last_vwap = df["vwap"].iloc[-1]
        cmp = df["close"].iloc[-1]
        return cmp < last_vwap
    except Exception as e:
        print(f"⚠️ VWAP cross check failed for {symbol}: {e}")
        return False

# ✅ ✅ ✅ BULLISH PIVOT DETECTION ✅ ✅ ✅

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
