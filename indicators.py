# indicators.py

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
import pandas_ta as pta

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

def check_supertrend_flip(kite, symbol, period=10, multiplier=3):
    """
    Returns True if Supertrend has flipped to bearish.
    """
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
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            length=period,
            multiplier=multiplier
        )
        # Auto-detect trend column
        trend_col = [c for c in supertrend.columns if "SUPERT" in c and "d" in c][0]
        last_trend = supertrend[trend_col].iloc[-1]
        return last_trend == -1
    except Exception as e:
        print(f"⚠️ Supertrend check failed for {symbol}: {e}")
        return False

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

def calculate_atr_trailing_sl(kite, symbol, cmp, atr_multiplier=1.5):
    """
    Computes ATR-based trailing stoploss price.
    """
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
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            window=14
        ).average_true_range().iloc[-1]
        trailing_sl = round(cmp - atr * atr_multiplier, 2)
        return trailing_sl
    except Exception as e:
        print(f"⚠️ ATR trailing SL error for {symbol}: {e}")
        return None

def check_rsi_bearish_divergence(kite, symbol, lookback=5):
    """
    Returns True if bearish RSI divergence is detected.
    """
    try:
        instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
        hist = kite.historical_data(
            instrument_token=instrument_token,
            from_date=pd.Timestamp.today() - pd.Timedelta(days=30),
            to_date=pd.Timestamp.today(),
            interval="day"
        )
        df = pd.DataFrame(hist)
        df["rsi"] = RSIIndicator(df["Close"], window=14).rsi()

        if len(df) < lookback + 1:
            return False

        price_highs = df["Close"].iloc[-lookback:]
        rsi_values = df["rsi"].iloc[-lookback:]

        if price_highs.is_monotonic_increasing and rsi_values.is_monotonic_decreasing:
            return True

        return False
    except Exception as e:
        print(f"⚠️ RSI divergence check failed for {symbol}: {e}")
        return False

def check_vwap_cross(kite, symbol):
    """
    Returns True if CMP < VWAP of last 10 days.
    """
    try:
        instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
        hist = kite.historical_data(
            instrument_token=instrument_token,
            from_date=pd.Timestamp.today() - pd.Timedelta(days=15),
            to_date=pd.Timestamp.today(),
            interval="day"
        )
        df = pd.DataFrame(hist)
        df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["cum_tp_vol"] = (df["typical_price"] * df["Volume"]).cumsum()
        df["cum_vol"] = df["Volume"].cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"]
        last_vwap = df["vwap"].iloc[-1]
        cmp = df["Close"].iloc[-1]
        return cmp < last_vwap
    except Exception as e:
        print(f"⚠️ VWAP cross check failed for {symbol}: {e}")
        return False
