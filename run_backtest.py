# run_backtest.py

import pandas as pd
import backtrader as bt
import os
from datetime import datetime

# === CONFIGURATION ===
DATA_FOLDER = '/root/falah-ai-bot/historical_data'

class SimpleStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(period=14)

    def next(self):
        if self.rsi[0] < 35:
            self.buy()
        elif self.rsi[0] > 70:
            self.sell()

# === UTILITY FUNCTIONS ===
def is_valid_file(file_path):
    try:
        df = pd.read_csv(file_path, nrows=5)
        required_columns = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(set(df.columns.str.lower())):
            print(f"⚠️ Skipping {os.path.basename(file_path)}: Missing required columns")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Skipping {os.path.basename(file_path)}: Error reading CSV -> {e}")
        return False

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df.set_index('datetime', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df

# === MAIN EXECUTION ===
cerebro = bt.Cerebro()

symbols = [f[:-4] for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
print(f"✅ Total symbols found: {len(symbols)}")
valid_count = 0

for symbol in symbols:
    file_path = os.path.join(DATA_FOLDER, f"{symbol}.csv")
    if not is_valid_file(file_path):
        continue

    df = load_data(file_path)
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data, name=symbol)
    valid_count += 1

print(f"✅ Total valid symbols loaded: {valid_count}")
cerebro.addstrategy(SimpleStrategy)
cerebro.run()

print("===== BACKTEST COMPLETE =====")
print(f"Attempted symbols: {len(symbols)}")
print(f"Valid symbols processed: {valid_count}")
