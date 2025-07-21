# generate_training_data.py

import os
import pandas as pd
import numpy as np
import pandas_ta as ta

INPUT_DIR = "/root/falah-ai-bot/historical_data"
OUTPUT_FILE = "/root/falah-ai-bot/your_training_data.csv"

rows = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    symbol = file.replace(".csv", "")
    path = os.path.join(INPUT_DIR, file)

    df = pd.read_csv(path)
    if len(df) < 30:
        continue

    # Indicators
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["EMA10"] = ta.ema(df["close"], length=10)
    df["EMA21"] = ta.ema(df["close"], length=21)
    df["VolumeChange"] = df["volume"].pct_change()

    # Target Outcome: Future high >= +3% within next 10 candles
    df["Future_High"] = df["high"].rolling(window=10, min_periods=1).max().shift(-1)
    df["Outcome"] = (df["Future_High"] >= df["close"] * 1.03).astype(int)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["RSI", "ATR", "ADX", "EMA10", "EMA21", "VolumeChange", "Outcome"])

    # Append clean rows
    for _, row in df.iterrows():
        rows.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "close": row["close"],
            "RSI": row["RSI"],
            "ATR": row["ATR"],
            "ADX": row["ADX"],
            "EMA10": row["EMA10"],
            "EMA21": row["EMA21"],
            "VolumeChange": row["VolumeChange"],
            "Outcome": row["Outcome"]
        })

# Final DataFrame
final_df = pd.DataFrame(rows)

# Save final training data
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Final training data saved to {OUTPUT_FILE} with {len(final_df)} rows.")
