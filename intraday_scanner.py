# intraday_scanner.py
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import pytz
import joblib

from kiteconnect import KiteConnect
from data_fetch import get_intraday_data  # Assumes you have 15min intraday fetcher
from ai_engine import extract_features   # Now available
from credentials import get_kite

# Constants
IST = pytz.timezone("Asia/Kolkata")
MODEL_PATH = "model.pkl"
FILTERED_FILE = "final_screened.json"
THRESHOLD = 0.25  # AI score threshold

def run_intraday_scan():
    # ✅ Load filtered stocks
    if not os.path.exists(FILTERED_FILE):
        print(f"❌ Filtered file not found: {FILTERED_FILE}")
        return pd.DataFrame()

    with open(FILTERED_FILE) as f:
        data = json.load(f)
    symbols = list(data.keys())
    print(f"🔍 Loaded {len(symbols)} symbols for intraday scan")

    # ✅ Load AI model
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ AI model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return pd.DataFrame()

    kite = get_kite()
    results = []

    for symbol in symbols:
        try:
            df = get_intraday_data(kite, symbol, interval="15minute", days=1)
            if df is None or len(df) < 21:
                continue

            features = extract_features(df)
            if features is None:
                continue

            X = pd.DataFrame([features])
            score = model.predict_proba(X)[0][1]  # Assuming binary model, class 1 = bullish

            if score >= THRESHOLD:
                results.append({
                    "Symbol": symbol,
                    "Score": round(score, 3),
                    "RSI": round(features["RSI"], 2),
                    "EMA10": round(features["EMA10"], 2),
                    "EMA21": round(features["EMA21"], 2),
                    "VolumeChange": round(features["VolumeChange"], 2),
                })

        except Exception as e:
            print(f"⚠️ {symbol}: {e}")
            continue

    if not results:
        print("⚠️ No stocks passed AI intraday filters")
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(by="Score", ascending=False)

# Debug run
if __name__ == "__main__":
    df = run_intraday_scan()
    print(df)
