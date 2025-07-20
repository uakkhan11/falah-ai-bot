# ai_engine.py

import pandas as pd
import joblib
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# Load your pre-trained AI model
try:
    model = joblib.load("model.pkl")
    print("✅ AI Model loaded successfully.")
except Exception as e:
    print(f"⚠️ AI Model load failed: {e}")
    model = None


def extract_features(df):
    """
    Extracts features for AI model from latest row:
    RSI (14), ATR (14), EMA10, EMA21, Volume Change %
    """
    if len(df) < 22:
        raise ValueError("Not enough data for feature extraction (minimum 22 rows required).")

    df = df.copy()

    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["atr"] = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    df["ema_10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
    df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["vol_change"] = df["volume"].pct_change() * 100

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
    Returns AI score (probability of positive class).
    """
    if model is None:
        print("⚠️ AI model not loaded. Default score = 0.0")
        return 0.0

    try:
        features = extract_features(df)
        input_df = pd.DataFrame([features], columns=["rsi", "atr", "ema_10", "ema_21", "vol_change"])
        score = model.predict_proba(input_df)[0][1]  # Class 1 probability
        return round(float(score), 4)
    except Exception as e:
        print(f"⚠️ AI score calculation failed: {e}")
        return 0.0


def get_ai_score_debug(df):
    """
    Returns AI score with detailed feature breakdown for debugging.
    """
    if model is None:
        print("⚠️ AI model not loaded. Debug score = 0.0")
        return 0.0, {}

    try:
        features = extract_features(df)
        input_df = pd.DataFrame([features], columns=["rsi", "atr", "ema_10", "ema_21", "vol_change"])
        score = model.predict_proba(input_df)[0][1]

        debug_info = {
            "RSI": round(features[0], 2),
            "ATR": round(features[1], 2),
            "EMA10": round(features[2], 2),
            "EMA21": round(features[3], 2),
            "VolChange%": round(features[4], 2),
            "AI_Score": round(score, 4)
        }

        return round(float(score), 4), debug_info
    except Exception as e:
        print(f"⚠️ AI debug score calculation failed: {e}")
        return 0.0, {}
