import os
import pandas as pd
import backtrader as bt
from datetime import datetime

HIST_DIR = "/root/falah-ai-bot/historical_data/"

class SimpleStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.ema10 = bt.indicators.EMA(self.data.close, period=10)
        self.ema21 = bt.indicators.EMA(self.data.close, period=21)

    def next(self):
        if not self.position:
            if self.rsi < 30 and self.ema10 > self.ema21:
                self.buy()
        else:
            if self.rsi > 70 or self.ema10 < self.ema21:
                self.close()

def load_csv_file(csv_file):
    # Quick header validation before loading the entire file
    try:
        df_head = pd.read_csv(csv_file, nrows=1)
        if 'date' not in df_head.columns:
            print(f"❌ Skipping {csv_file} (missing 'date' column)")
            return None
    except Exception as e:
        print(f"❌ Error reading {csv_file}: {e}")
        return None

    try:
        df = pd.read_csv(csv_file, parse_dates=["date"])
        df = df.dropna(subset=["date", "close", "open", "high", "low", "volume"])
        df = df.sort_values("date")
        return df
    except Exception as e:
        print(f"❌ Failed to process {csv_file}: {e}")
        return None

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SimpleStrategy)

    csv_files = [os.path.join(HIST_DIR, f) for f in os.listdir(HIST_DIR) if f.endswith(".csv")]
    print(f"✅ Found {len(csv_files)} CSV files.")

    valid_count = 0

    for csv_file in csv_files:
        df = load_csv_file(csv_file)
        if df is None or df.empty:
            continue

        data = bt.feeds.PandasData(
            dataname=df,
            datetime="date",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=None,
        )
        cerebro.adddata(data)
        valid_count += 1

    print(f"✅ Loaded {valid_count} valid CSV files into Backtrader.")

    if valid_count == 0:
        print("⚠️ No valid data files found. Exiting.")
        exit()

    cerebro.broker.set_cash(100000)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print("✅ Starting backtest...")
    results = cerebro.run()
    strat = results[0]

    print(f"🏁 Final Portfolio Value: ₹{cerebro.broker.getvalue():.2f}")

    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"📊 Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")

    trades = strat.analyzers.trades.get_analysis()
    print(f"📈 Total Trades: {trades.total.closed if 'total' in trades and 'closed' in trades.total else 'N/A'}")

    # Uncomment to visualize
    # cerebro.plot()
