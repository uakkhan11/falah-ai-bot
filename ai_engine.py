# ai_engine.py

import pandas as pd
from datetime import datetime
import pytz
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from data_fetch import fetch_recent_historical  # assumes already implemented

IST = pytz.timezone("Asia/Kolkata")

def compute_ai_score(df):
    """
    Compute a bullish entry score for the given stock data.
    Factors considered:
    - RSI between 35-65 or 65-70 with extra boost
    - EMA10 > EMA21
    - Volume surge (last volume > 1.3x average)
    - Green candle
    - Supertrend confirmation (if available)
    """
    if df is None or len(df) < 21:
        return 0, ["Insufficient data"]

    try:
        features = extract_features(df)
        if not features:
            return 0.0, ["Invalid features"]

        model = joblib.load("model.pkl")
        probs = model.predict_proba([list(features.values())])  # [[0.3, 0.7]]
        ai_score = float(probs[0][1])  # Positive class score

        reasons = []

    df = df.copy()

    df["EMA10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
    df["EMA21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["ATR"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    df["VolumeChange"] = df["volume"] / (df["volume"].rolling(10).mean() + 1e-9)

    last = df.iloc[-1]
    score = 0
    reasons = []

    if last["EMA10"] > last["EMA21"]:
        score += 10
        reasons.append("EMA10 > EMA21")

    if 35 <= last["RSI"] <= 65:
        score += 10
        reasons.append("RSI between 35-65")

    elif 65 < last["RSI"] <= 70:
        score += 5
        reasons.append("RSI 65-70")

    elif last["RSI"] > 70:
        score -= 10
        reasons.append("RSI > 70 (overbought)")

    if last["VolumeChange"] > 1.3:
        score += 10
        reasons.append("Volume spike")

    if last["close"] > last["open"]:
        score += 5
        reasons.append("Green candle")

    if "Supertrend" in df.columns and last["close"] > last["Supertrend"]:
        score += 5
        reasons.append("Supertrend green")

    return score, reasons


def calculate_ai_exit_score(stock_data, trailing_sl, current_price, atr_value=None):
    score = 0
    reasons = []

    if "ADX" not in stock_data.columns:
        adx_indicator = ADXIndicator(
            high=stock_data["high"],
            low=stock_data["low"],
            close=stock_data["close"],
            window=14
        )
        stock_data["ADX"] = adx_indicator.adx()

    last = stock_data.iloc[-1]

    volume_col = None
    for col in ["Volume", "volume"]:
        if col in stock_data.columns:
            volume_col = col
            break

    if volume_col:
        recent_vol = last[volume_col]
        avg_vol = stock_data[volume_col].tail(10).mean()
    else:
        recent_vol = avg_vol = None

    if current_price < trailing_sl:
        drop_pct = (trailing_sl - current_price) / trailing_sl
        penalty = int(drop_pct * 100)
        score -= penalty
        reasons.append(f"Trailing SL hit (-{penalty} pts)")

    if "Supertrend" in last and current_price < last["Supertrend"]:
        score -= 25
        reasons.append("Supertrend reversal")

    if recent_vol is not None and recent_vol < 0.7 * avg_vol:
        score -= 10
        reasons.append("Volume drop")

    if last["close"] < last["open"] and (last["open"] - last["close"]) > 0.005 * last["open"]:
        score -= 15
        reasons.append("Bearish candle")

    if recent_vol is not None and recent_vol > 1.5 * avg_vol and last["close"] < last["open"]:
        score -= 15
        reasons.append("High-volume bearish candle")

    now = datetime.now(IST)
    if now.hour == 15:
        score -= 20
        reasons.append("End of day exit window")

    if last["close"] > last["open"] and (last["close"] - last["open"]) > 0.01 * last["open"]:
        score += 10
        reasons.append("Strong bullish close")

    adx_col = "ADX_14" if "ADX_14" in stock_data.columns else ("ADX" if "ADX" in stock_data.columns else None)
    if adx_col and stock_data[adx_col].iloc[-1] < 20:
        score -= 10
        reasons.append("Weak trend (low ADX)")

    if atr_value:
        atr_distance = abs(current_price - trailing_sl) / atr_value
        if atr_distance < 0.5:
            score -= 10
            reasons.append("Price near trailing SL")

    if len(stock_data) >= 2:
        gap_pct = (last["open"] - stock_data["close"].iloc[-2]) / stock_data["close"].iloc[-2]
        if gap_pct < -0.01:
            score -= 10
            reasons.append("Gap down open")

    return score, reasons


def analyze_exit_signals(kite, symbol, avg_price, current_price):
    try:
        stock_data = fetch_recent_historical(kite, symbol, days=30)
        if len(stock_data) < 15:
            return ["Insufficient data"]

        atr_value = stock_data["ATR"].iloc[-1] if "ATR" in stock_data.columns else None
        trailing_sl = avg_price + (current_price - avg_price) * 0.5

        score, reasons = calculate_ai_exit_score(stock_data, trailing_sl, current_price, atr_value=atr_value)

        if score < -20:
            return reasons
        return []
    except Exception as e:
        return [f"Error: {e}"]


def extract_features(df):
    if len(df) < 21:
        return None

    df = df.copy()

    df["EMA10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
    df["EMA21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["ATR"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["VolumeChange"] = df["volume"] / (df["volume"].rolling(10).mean() + 1e-9)

    last = df.iloc[-1]

    return {
        "RSI": last["RSI"],
        "EMA10": last["EMA10"],
        "EMA21": last["EMA21"],
        "ATR": last["ATR"],
        "VolumeChange": last["VolumeChange"],
    }
