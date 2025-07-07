import os
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

HIST_DIR = "/root/falah-ai-bot/historical_data/"
out_rows = []

for file in os.listdir(HIST_DIR):
    if not file.endswith(".csv"):
        continue
    sym = file.replace(".csv", "")
    df = pd.read_csv(os.path.join(HIST_DIR, file))

    if len(df) < 21:
        continue

    df["SMA20"] = df["close"].rolling(20).mean()
    df["EMA10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
    df["EMA21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
    df["ATR"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["VolumeChange"] = df["volume"] / df["volume"].rolling(10).mean()

    df = df.dropna()

    # Create Target column: did next day's close increase >1.5%?
    df["Target"] = (
        df["close"].shift(-1) > df["close"] * 1.015
    ).astype(int)

    for i, row in df.iterrows():
        out_rows.append({
            "Date": row["date"],
            "Symbol": sym,
            "RSI": row["RSI"],
            "EMA10": row["EMA10"],
            "EMA21": row["EMA21"],
            "SMA20": row["SMA20"],
            "ATR": row["ATR"],
            "VolumeChange": row["VolumeChange"],
            "Target": row["Target"],
        })

final_df = pd.DataFrame(out_rows)
final_df.to_csv("training_data.csv", index=False)
print(f"✅ Training dataset created: {len(final_df)} rows")
