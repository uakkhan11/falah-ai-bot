# run_backtest.py

import os
import pandas as pd
import backtrader as bt

class DebugAlwaysBuy(bt.Strategy):
    def __init__(self):
        pass

    def next(self):
        if not self.position:
            self.buy()
            print(f"✅ BUY executed at {self.data.datetime.date(0)} | Close={self.data.close[0]:.2f}")

class DebugSymbolCheck(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=14)

    def next(self):
        rsi_val = self.rsi[0]
        if not self.position:
            if rsi_val > 35 and rsi_val < 70:
                self.buy()
                print(f"✅ BUY {self.data._name} at {self.data.datetime.date(0)} | Close={self.data.close[0]:.2f} | RSI={rsi_val:.2f}")
            else:
                print(f"⏩ SKIP {self.data._name} | {self.data.datetime.date(0)} | RSI={rsi_val:.2f}")

# Load symbols from historical_data directory
DATA_DIR = '/root/falah-ai-bot/historical_data'
files = os.listdir(DATA_DIR)
symbols = [f.replace('.csv', '') for f in files if f.endswith('.csv')]
print(f"✅ Total symbols found: {len(symbols)}")

valid_symbols = []

cerebro = bt.Cerebro()

for symbol in symbols:
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    try:
        df = pd.read_csv(file_path)

        if not {'date', 'open', 'high', 'low', 'close', 'volume'}.issubset(df.columns):
            print(f"⚠️ Skipping {symbol}: Missing required columns")
            continue

        df.rename(columns={'date': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        if len(df) < 100:
            print(f"⚠️ Skipping {symbol}: Insufficient data ({len(df)} rows)")
            continue

        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name=symbol)
        valid_symbols.append(symbol)

    except Exception as e:
        print(f"⚠️ Skipping {symbol}: Error loading data -> {e}")

print(f"✅ Total valid symbols loaded: {len(valid_symbols)}")

# Choose strategy (Always-buy for sanity)
cerebro.addstrategy(DebugAlwaysBuy)

cerebro.broker.set_cash(100000)
cerebro.run()

print("===== BACKTEST COMPLETE =====")
print(f"Attempted symbols: {len(symbols)}")
print(f"Valid symbols processed: {len(valid_symbols)}")

