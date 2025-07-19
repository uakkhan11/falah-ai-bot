# run_backtest.py

import os
import pandas as pd
import backtrader as bt

class DummyStrategy(bt.Strategy):
    def __init__(self):
        pass

    def next(self):
        pass

# Path to historical_data directory
DATA_DIR = "/root/falah-ai-bot/historical_data/"

symbols = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
print(f"✅ Total symbols found: {len(symbols)}")

valid_symbols = 0
cerebro = bt.Cerebro()

for symbol_file in symbols:
    file_path = os.path.join(DATA_DIR, symbol_file)
    try:
        df = pd.read_csv(file_path)

        # Fix column name
        if 'date' in df.columns:
            df.rename(columns={'date': 'datetime'}, inplace=True)

        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print(f"⚠️ Skipping {symbol_file}: Missing required columns")
            continue

        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df.dropna(subset=['datetime'], inplace=True)

        df = df.sort_values('datetime')
        df.set_index('datetime', inplace=True)

        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name=symbol_file.replace('.csv', ''))

        valid_symbols += 1

    except Exception as e:
        print(f"⚠️ Skipping {symbol_file}: Error -> {e}")

print(f"✅ Total valid symbols loaded: {valid_symbols}")

if valid_symbols == 0:
    print("❌ No valid data to backtest.")
else:
    cerebro.addstrategy(DummyStrategy)
    cerebro.run()
    print("===== BACKTEST COMPLETE =====")
