import os
import pandas as pd
from datetime import datetime
import pytz
import joblib

from kiteconnect import KiteConnect
from data_fetch import get_intraday_data
from ai_engine import extract_features
from indicators import calculate_rsi, calculate_ema, detect_macd_bullish_cross, detect_supertrend_green
from credentials import get_kite, load_secrets
from live_price_reader import get_symbol_price_map
from holdings import get_existing_holdings
import gspread

# Constants
IST = pytz.timezone("Asia/Kolkata")
MODEL_PATH = "model.pkl"
VOLUME_SURGE_THRESHOLD = 2.0
THRESHOLD = 0.25

def get_halal_list(sheet_key):
    gc = gspread.service_account(filename="/root/falah-ai-bot/falah-credentials.json")
    sheet = gc.open_by_key(sheet_key)
    ws = sheet.worksheet("HalalList")
    symbols = ws.col_values(1)[1:]
    return [s.strip().upper() for s in symbols if s.strip()]

def run_intraday_scan():
    debug_logs = []
    results = []

    # ✅ Load Halal list
    secrets = load_secrets()
    symbols = get_halal_list(secrets["google"]["spreadsheet_key"])
    holdings = get_existing_holdings()
    live_prices = get_symbol_price_map()

    debug_logs.append(f"🔍 Loaded {len(symbols)} symbols from Halal list")

    if not live_prices:
        msg = "⚠️ No live prices available. Possibly market is closed."
        print(msg)
        return pd.DataFrame(), [msg]

    # ✅ Load model
    try:
        model = joblib.load(MODEL_PATH)
        debug_logs.append("✅ AI model loaded")
    except Exception as e:
        msg = f"❌ Failed to load model: {e}"
        return pd.DataFrame(), [msg]

    kite = get_kite()

    for symbol in sorted(set(symbols)):
        if symbol in holdings:
            debug_logs.append(f"⏭ {symbol}: Skipped (already in holdings)")
            continue
        if symbol not in live_prices:
            debug_logs.append(f"⏭ {symbol}: Skipped (no live price)")
            continue

        try:
            df = get_intraday_data(kite, symbol, interval="15minute", days=1)
            if df is None or len(df) < 21:
                debug_logs.append(f"⏭ {symbol}: insufficient data")
                continue

            df["rsi"] = calculate_rsi(df["close"])
            df["ema10"] = calculate_ema(df["close"], 10)
            df["ema21"] = calculate_ema(df["close"], 21)

            rsi = df["rsi"].iloc[-1]
            ema10 = df["ema10"].iloc[-1]
            ema21 = df["ema21"].iloc[-1]
            volume_today = df["volume"].iloc[-1]
            volume_avg = df["volume"].iloc[-6:-1].mean()
            volume_ratio = volume_today / volume_avg if volume_avg else 0

            if not (ema10 > ema21):
                continue
            if not (35 <= rsi <= 65):
                continue
            if not detect_macd_bullish_cross(df):
                continue
            if not detect_supertrend_green(df):
                continue
            if volume_ratio < VOLUME_SURGE_THRESHOLD:
                continue

            features = extract_features(df)
            if features is None:
                continue

            X = pd.DataFrame([features])[['rsi', 'ema10', 'ema21', 'atr', 'adx', 'volumechange']]
            score = model.predict_proba(X)[0][1]

            if score >= THRESHOLD:
                results.append({
                    "symbol": symbol,
                    "Score": round(score, 3),
                    "rsi": round(rsi, 2),
                    "ema10": round(ema10, 2),
                    "ema21": round(ema21, 2),
                    "volume_ratio": round(volume_ratio, 2),
                    "ai_reasons": features.get("ai_reasons", "N/A")
                })
                debug_logs.append(f"✅ {symbol} passed with Score {round(score, 3)}")

        except Exception as e:
            debug_logs.append(f"⚠️ {symbol}: {e}")

    if not results:
        debug_logs.append("⚠️ No stocks passed the strategy")

    df_result = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    return df_result, debug_logs

if __name__ == "__main__":
    df, logs = run_intraday_scan()
    print("\n📈 Final Intraday Picks:\n", df)
    print("\n🧾 Debug Logs:")
    for log in logs:
        print(log)
