# run_backtest.py

import os
import pandas as pd
import backtrader as bt
from datetime import datetime

class SimpleStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=14)

    def next(self):
        if not self.position and self.rsi < 30:
            self.buy()
        elif self.position and self.rsi > 70:
            self.sell()

historical_data_path = '/root/falah-ai-bot/historical_data'
files = [f for f in os.listdir(historical_data_path) if f.endswith('.csv')]

print(f"✅ Total symbols loaded from historical_data: {len(files)}")

symbols_attempted = 0
symbols_valid = 0

cerebro = bt.Cerebro()
cerebro.broker.set_cash(1000000)

for file in files:
    symbols_attempted += 1
    symbol = file.replace('.csv', '')
    file_path = os.path.join(historical_data_path, file)

    try:
        cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = pd.read_csv(file_path, names=cols, header=None)

        df['datetime'] = pd.to_datetime(df['datetime'])

        if df.empty or 'datetime' not in df.columns:
            print(f"⚠️ Skipping {symbol}: Invalid or empty data")
            continue

        df.set_index('datetime', inplace=True)
        data = bt.feeds.PandasData(dataname=df)

        cerebro.adddata(data, name=symbol)
        symbols_valid += 1

    except Exception as e:
        print(f"⚠️ Skipping {symbol}: Error reading CSV -> {e}")
        continue

print(f"===== FINAL SUMMARY =====")
print(f"Total Symbols Attempted: {symbols_attempted}")
print(f"Valid Symbols Backtested: {symbols_valid}")

results = cerebro.run()
cerebro.broker.get_value()
print(f"===== END =====")
