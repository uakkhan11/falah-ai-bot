import os
import pandas as pd

HIST_DIR = "/root/falah-ai-bot/historical_data/"

symbols = [f for f in os.listdir(HIST_DIR) if f.endswith(".csv")]

summary = []
for symbol_file in symbols:
    symbol = symbol_file.replace(".csv", "")
    try:
        df = pd.read_csv(os.path.join(HIST_DIR, symbol_file), parse_dates=['date'])
        if len(df) < 50 or df.isnull().values.any():
            print(f"⚠️ Skipping {symbol} due to insufficient data.")
            continue
        df = df.sort_values('date')
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        min_rsi = df['rsi'].min()
        max_rsi = df['rsi'].max()
        print(f"{symbol}: Min RSI={min_rsi:.2f}, Max RSI={max_rsi:.2f}")
        summary.append({"symbol": symbol, "min_rsi": min_rsi, "max_rsi": max_rsi})
    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")

summary_df = pd.DataFrame(summary)
summary_df.to_csv("rsi_min_max_summary.csv", index=False)
print("✅ RSI min/max summary saved to rsi_min_max_summary.csv")
