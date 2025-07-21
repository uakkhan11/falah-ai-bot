# generate_training_data.py

import os
import pandas as pd
import numpy as np
import pandas_ta as ta

INPUT_DIR = "/root/falah-ai-bot/historical_data"
OUTPUT_FILE = "/root/falah-ai-bot/training_data.csv"

rows = []

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    symbol = file.replace(".csv", "")
    path = os.path.join(INPUT_DIR, file)

    df = pd.read_csv(path)

    # ✅ Column Check
    required_cols = {"close", "high", "low"}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ Skipping {symbol}: Missing columns {required_cols - set(df.columns)}")
        continue

    if len(df) < 30:
        print(f"⚠️ Skipping {symbol}: Not enough data rows ({len(df)} rows).")
        continue

    # ✅ Indicators
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

    # ✅ Target Label: > +2% in next 5 days
    df["future_close"] = df["close"].shift(-5)
    df["future_return_pct"] = (df["future_close"] - df["close"]) / df["close"] * 100
    df["Target"] = (df["future_return_pct"] > 2).astype(int)

    # ✅ Drop NaNs after indicators
    df = df.dropna(subset=["RSI", "ATR", "ADX", "Target"])

    for _, row in df.iterrows():
        rows.append({
            "Symbol": symbol,
            "RSI": row["RSI"],
            "ATR": row["ATR"],
            "ADX": row["ADX"],
            "Target": row["Target"]
        })

# ✅ Final DataFrame
final_df = pd.DataFrame(rows)
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(final_df)} rows to {OUTPUT_FILE}")
