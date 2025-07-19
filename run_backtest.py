import os
import pandas as pd
import backtrader as bt
from indicators import add_indicators
from ai_engine import get_ai_score

DATA_DIR = "/root/falah-ai-bot/historical_data/"
large_mid_cap_file = "large_mid_cap.json"

# Load Large and Mid Cap symbols
import json

with open('large_mid_cap.json') as f:
    large_mid_symbols = set(json.load(f))

class AIStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=14)
        self.ema10 = bt.indicators.EMA(self.data.close, period=10)
        self.ema21 = bt.indicators.EMA(self.data.close, period=21)

    def next(self):
        if not self.position:
            if (self.rsi[0] > 35 and self.rsi[0] < 65) and (self.ema10[0] > self.ema21[0]):
                ai_score = get_ai_score(self.data._dataname)
                if ai_score >= 0.25:
                    self.buy()
                    print(f"{self.data._name}: BUY at {self.data.close[0]:.2f}, RSI={self.rsi[0]:.2f}, AI Score={ai_score:.2f}")
        else:
            if self.rsi[0] > 70:
                self.close()
                print(f"{self.data._name}: SELL at {self.data.close[0]:.2f}, RSI={self.rsi[0]:.2f}")

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    total_symbols = 0
    executed_trades = 0

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"):
            continue
        symbol = file.replace(".csv", "")
        if symbol not in large_mid_symbols:
            continue

        file_path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv(file_path, parse_dates=["datetime"])
            df = df.dropna()
            if len(df) < 100:
                print(f"Skipping {symbol}: insufficient data ({len(df)} rows)")
                continue
            df = add_indicators(df)

            data = bt.feeds.PandasData(dataname=df, datetime="datetime", open="open", high="high",
                                       low="low", close="close", volume="volume", openinterest=None)
            cerebro.adddata(data, name=symbol)
            total_symbols += 1
        except Exception as e:
            print(f"Skipping {symbol}: {e}")
            continue

    cerebro.addstrategy(AIStrategy)
    print(f"Total symbols loaded: {total_symbols}")
    results = cerebro.run()
    print("Backtest complete.")
