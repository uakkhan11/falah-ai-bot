# run_backtest.py

import os
import pandas as pd
import backtrader as bt
from datetime import datetime

# ===== Strategy Placeholder =====
class AIStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema10 = bt.indicators.EMA(self.data.close, period=10)
        self.ema21 = bt.indicators.EMA(self.data.close, period=21)

    def next(self):
        if self.position:
            if self.rsi < 40:
                self.close()
        else:
            if 35 < self.rsi < 65 and self.ema10 > self.ema21:
                self.buy()

# ===== Backtest Runner =====
historical_data_folder = './historical_data'
csv_files = [f for f in os.listdir(historical_data_folder) if f.endswith('.csv')]
symbols = [os.path.splitext(f)[0] for f in csv_files]

print(f"✅ Total symbols loaded from historical_data: {len(symbols)}")

total_trades = 0
valid_symbols = 0

for symbol in symbols:
    file_path = os.path.join(historical_data_folder, f"{symbol}.csv")

    try:
        df = pd.read_csv(file_path)
        if 'datetime' not in df.columns:
            print(f"⚠️ Skipping {symbol}: Missing datetime column")
            continue
        df['datetime'] = pd.to_datetime(df['datetime'])
    except Exception as e:
        print(f"⚠️ Skipping {symbol}: Error loading CSV -> {e}")
        continue

    if df.empty or len(df) < 50:
        print(f"⚠️ Skipping {symbol}: Not enough data ({len(df)} rows)")
        continue

    df.dropna(inplace=True)

    data = bt.feeds.PandasData(
        dataname=df,
        datetime='datetime',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )

    cerebro = bt.Cerebro()
    cerebro.addstrategy(AIStrategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(100000)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    print(f"▶️ Running backtest for {symbol}")
    initial_value = cerebro.broker.getvalue()
    cerebro.run()
    final_value = cerebro.broker.getvalue()

    valid_symbols += 1

    if final_value != initial_value:
        total_trades += 1
        print(f"✅ {symbol} - PnL: {round(final_value - initial_value, 2)}")
    else:
        print(f"❌ {symbol} - No trades executed")

print("\n===== FINAL SUMMARY =====")
print(f"Total Symbols Attempted: {len(symbols)}")
print(f"Valid Symbols Backtested: {valid_symbols}")
print(f"Total Trades Executed: {total_trades}")
print("===== END =====")
