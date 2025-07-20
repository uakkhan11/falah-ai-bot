# ai_engine.py

import pandas as pd
import joblib
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange


# Load your pre-trained AI model (ensure model.pkl exists in your directory)
try:
    model = joblib.load("model.pkl")
except Exception as e:
    print(f"⚠️ AI Model load failed: {e}")
    model = None


def extract_features(df):
    """
    Extracts features required by the AI model from the latest data.
    Features: RSI, ATR, EMA10, EMA21, Volume Change %
    """
    if len(df) < 22:
        raise ValueError("Not enough data for feature extraction (minimum 22 rows required).")

    df = df.copy()

    # RSI 14
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()

    # ATR 14
    df["atr"] = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    ).average_true_range()

    # EMA 10 and EMA 21
    df["ema_10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
    df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()

    # Volume Change %
    df["vol_change"] = df["volume"].pct_change() * 100

    # Extract latest row features
    last_row = df.iloc[-1]
    features = [
        last_row["rsi"],
        last_row["atr"],
        last_row["ema_10"],
        last_row["ema_21"],
        last_row["vol_change"]
    ]

    return features


def get_ai_score(df):
    """
    Returns AI score (probability) for the latest candle.
    """
    if model is None:
        print("⚠️ AI model not loaded, returning score 0")
        return 0.0

    try:
        features = extract_features(df)
        input_data = pd.DataFrame([features], columns=["rsi", "atr", "ema_10", "ema_21", "vol_change"])
        score = model.predict_proba(input_data)[0][1]  # Assuming binary classifier, take probability of class 1
        return float(score)
    except Exception as e:
        print(f"⚠️ AI score calculation failed: {e}")
        return 0.0


def get_ai_score_debug(df):
    """
    Returns AI score with debug information for inspection.
    """
    if model is None:
        return 0.0, {}

    try:
        features = extract_features(df)
        input_data = pd.DataFrame([features], columns=["rsi", "atr", "ema_10", "ema_21", "vol_change"])
        score = model.predict_proba(input_data)[0][1]
        debug_info = {
            "RSI": round(features[0], 2),
            "ATR": round(features[1], 2),
            "EMA10": round(features[2], 2),
            "EMA21": round(features[3], 2),
            "VolChange%": round(features[4], 2),
            "AI_Score": round(score, 4)
        }
        return float(score), debug_info
    except Exception as e:
        print(f"⚠️ AI debug score calculation failed: {e}")
        return 0.0, {}
