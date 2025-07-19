import pandas as pd
import os
import json
import joblib
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

HIST_DIR = "/root/falah-ai-bot/historical_data/"
LARGE_MID_CAP_FILE = "/root/falah-ai-bot/large_mid_cap.json"
MODEL_PATH = "/root/falah-ai-bot/model.pkl"

model = joblib.load(MODEL_PATH)

def load_large_mid_cap_symbols():
    with open(LARGE_MID_CAP_FILE) as f:
        symbols = json.load(f)
    print(f"✅ Loaded {len(symbols)} Large/Mid Cap symbols.")
    return set(symbols)

def process_symbol(symbol):
    file_path = os.path.join(HIST_DIR, f"{symbol}.csv")
    if not os.path.exists(file_path):
        return []

    df = pd.read_csv(file_path)
    if len(df) < 300:
        return []

    df["EMA10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
    df["EMA21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["ATR"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["VolumeChange"] = df["volume"] / df["volume"].rolling(10).mean()

    df = df.dropna()

    scores = []
    for idx, row in df.iterrows():
        features = pd.DataFrame([{
            "RSI": row["RSI"],
            "EMA10": row["EMA10"],
            "EMA21": row["EMA21"],
            "ATR": row["ATR"],
            "VolumeChange": row["VolumeChange"]
        }])
        proba = model.predict_proba(features)[0][1]
        scores.append(proba)
    return scores

def main():
    symbols = load_large_mid_cap_symbols()
    all_scores = []

    for idx, symbol in enumerate(symbols):
        scores = process_symbol(symbol)
        all_scores.extend(scores)
        print(f"[{idx+1}/{len(symbols)}] {symbol}: {len(scores)} scores processed.")

    all_scores = [round(s, 4) for s in all_scores]
    df = pd.DataFrame({"Score": all_scores})
    df.to_csv("ai_score_distribution.csv", index=False)
    print(f"\n✅ Completed. Total {len(all_scores)} scores saved to ai_score_distribution.csv")

    print("📊 Score Distribution:")
    print(df.describe())

if __name__ == "__main__":
    main()
