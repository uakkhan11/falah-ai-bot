# generate_training_data.py

import os
import pandas as pd
import numpy as np
import pandas_ta as ta

INPUT_DIR = "/root/falah-ai-bot/historical_data"
OUTPUT_FILE = "/root/falah-ai-bot/training_data.csv"

rows = []

# Loop over all CSV files
for file in os.listdir(INPUT_DIR):
    if not file.endswith(".csv"):
        continue

    symbol = file.replace(".csv", "")
    path = os.path.join(INPUT_DIR, file)

    df = pd.read_csv(path)
    if len(df) < 30:
        continue

    # Compute indicators
    df["RSI"] = ta.rsi(df["close"], length=14)
    df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ADX"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]

    # Shift close by -5 days to get future return
    df["future_close"] = df["close"].shift(-5)
    df["future_return_pct"] = (df["future_close"] - df["close"]) / df["close"] * 100

    # Target: 1 if price goes up more than +2% in 5 days
    df["Target"] = np.where(df["future_return_pct"] > 2, 1, 0)

    # Drop NaN rows (due to indicators and shifting)
    df = df.dropna()

    # For AI_Score placeholder, random between 0.4 and 0.9
    df["AI_Score"] = np.random.uniform(0.4, 0.9, size=len(df))

    for _, row in df.iterrows():
        rows.append({
            "Symbol": symbol,
            "RSI": row["RSI"],
            "ATR": row["ATR"],
            "ADX": row["ADX"],
            "AI_Score": row["AI_Score"],
            "Target": row["Target"]
        })

# Build dataframe
final_df = pd.DataFrame(rows)

# Save
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Training data saved to {OUTPUT_FILE} ({len(final_df)} rows).")
