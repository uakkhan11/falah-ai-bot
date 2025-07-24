# indicators.py

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
import pandas_ta as pta


def calculate_rsi(df, period=14):
    df['RSI'] = RSIIndicator(close=df['close'], window=period).rsi()
    return df


def calculate_ema(df, period=21, col='close'):
    df[f'EMA{period}'] = EMAIndicator(close=df[col], window=period).ema_indicator()
    return df


def calculate_macd(df):
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    return df


def calculate_vwap(df):
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (df['TP'] * df['volume']).cumsum() / df['volume'].cumsum()
    return df.drop(columns=['TP'])


def calculate_atr(df, period=14):
    df['ATR'] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=period).average_true_range()
    return df


def calculate_supertrend(df, period=10, multiplier=3):
    st = pta.supertrend(df["high"], df["low"], df["close"], length=period, multiplier=multiplier)
    trend_col = [col for col in st.columns if 'SUPERTd' in col][0]
    return st[trend_col]


def detect_supertrend_green(df):
    st_dir = calculate_supertrend(df)
    return st_dir.iloc[-1] == 1 if len(st_dir) > 0 else False


def detect_breakout(df, threshold=1.02):
    if len(df) < 2:
        return False, None
    prev_high = df["high"].iloc[-2]
    curr_close = df["close"].iloc[-1]
    breakout = curr_close > prev_high * threshold
    return breakout, round(prev_high * threshold, 2)


def detect_3green_days(df):
    if len(df) < 3:
        return False
    last3 = df.iloc[-3:]
    return all(last3["close"] > last3["open"])


def detect_darvas_box(df, lookback=20):
    if len(df) < lookback + 1:
        return False, None
    recent_high = df["high"].iloc[-lookback:-1].max()
    curr_close = df["close"].iloc[-1]
    breakout = curr_close > recent_high
    return breakout, round(recent_high, 2)


def detect_bullish_pivot(df, lookback=5):
    if len(df) < lookback + 2:
        return False
    pivot_lows = []
    for i in range(lookback, len(df) - 1):
        if df["low"].iloc[i] < df["low"].iloc[i - 1] and df["low"].iloc[i] < df["low"].iloc[i + 1]:
            pivot_lows.append(df["low"].iloc[i])
    return len(pivot_lows) >= 2 and pivot_lows[-1] > pivot_lows[-2]


def detect_recent_high(df, lookback=20):
    if len(df) < lookback:
        return None
    return df["close"].iloc[-lookback:].max()


def detect_recent_swing_low(df, lookback=5):
    if len(df) < lookback:
        return None
    return df["low"].iloc[-lookback:].min()


def calculate_chandelier_exit(df, atr_multiplier=3):
    atr = df['ATR'].iloc[-1] if 'ATR' in df else calculate_atr(df)['ATR'].iloc[-1]
    highest_close = df["close"].rolling(window=22).max().iloc[-1]
    return round(highest_close - atr * atr_multiplier, 2)


def get_best_trailing_sl(df, cmp, atr_multiplier=1.5, lookback=5):
    if 'ATR' not in df:
        df = calculate_atr(df)
    atr = df['ATR'].iloc[-1]
    recent_low = detect_recent_swing_low(df, lookback)
    chandelier_exit = calculate_chandelier_exit(df, atr_multiplier=3)
    atr_sl = cmp - atr * atr_multiplier
    valid_sl = [sl for sl in [recent_low, chandelier_exit, atr_sl] if sl is not None]
    return round(max(valid_sl), 2) if valid_sl else round(atr_sl, 2)


def add_all_indicators(df):
    df = calculate_rsi(df)
    df = calculate_ema(df, 10)
    df = calculate_ema(df, 21)
    df = calculate_macd(df)
    df = calculate_vwap(df)
    df = calculate_atr(df)
    return df
