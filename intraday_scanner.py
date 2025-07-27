# intraday_scanner.py
import os
import json
import pandas as pd
from datetime import datetime
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
    debug_logs = []

    # ✅ Load filtered stocks
    if not os.path.exists(FILTERED_FILE):
        msg = f"❌ Filtered file not found: {FILTERED_FILE}"
        print(msg)
        debug_logs.append(msg)
        return pd.DataFrame(), debug_logs

    with open(FILTERED_FILE) as f:
        data = json.load(f)

    # ✅ Handle different formats
    if isinstance(data, dict):
        symbols = list(data.keys())
    elif isinstance(data, list):
        symbols = [d.get("symbol") for d in data if isinstance(d, dict) and "symbol" in d]
    else:
        msg = f"❌ Unsupported format in {FILTERED_FILE}"
        print(msg)
        debug_logs.append(msg)
        return pd.DataFrame(), debug_logs

    print(f"🔍 Loaded {len(symbols)} symbols for intraday scan")
    debug_logs.append(f"Loaded {len(symbols)} symbols")

    # ✅ Load AI model
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ AI model loaded")
        debug_logs.append("AI model loaded")
    except Exception as e:
        msg = f"❌ Failed to load model: {e}"
        print(msg)
        debug_logs.append(msg)
        return pd.DataFrame(), debug_logs

    kite = get_kite()
    results = []

    for symbol in symbols:
        try:
            df = get_intraday_data(kite, symbol, interval="15minute", days=1)
            if df is None or len(df) < 21:
                debug_logs.append(f"⏭ Skipped {symbol}: insufficient data")
                continue

            features = extract_features(df)
            if features is None:
                debug_logs.append(f"⏭ Skipped {symbol}: feature extraction failed")
                continue

            X = pd.DataFrame([features])[['RSI', 'EMA10', 'EMA21', 'ATR', 'ADX', 'VolumeChange']]
            score = model.predict_proba(X)[0][1]

            if score >= THRESHOLD:
                results.append({
                    "Symbol": symbol,
                    "Score": round(score, 3),
                    "RSI": round(features.get("RSI", 0), 2),
                    "EMA10": round(features.get("EMA10", 0), 2),
                    "EMA21": round(features.get("EMA21", 0), 2),
                    "VolumeChange": round(features.get("VolumeChange", 0), 2),
                })
                debug_logs.append(f"✅ {symbol} passed with score {round(score,3)}")
            else:
                debug_logs.append(f"❌ {symbol} score {round(score,3)} below threshold")

        except Exception as e:
            msg = f"⚠️ {symbol}: {e}"
            print(msg)
            debug_logs.append(msg)
            continue

    if not results:
        msg = "⚠️ No stocks passed AI intraday filters"
        print(msg)
        debug_logs.append(msg)
        return pd.DataFrame(), debug_logs

    df_result = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    return df_result, debug_logs

# Debug run
if __name__ == "__main__":
    df, logs = run_intraday_scan()
    print(df)
    for log in logs:
        print(log)
