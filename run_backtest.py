import os
import pandas as pd
import backtrader as bt
from datetime import datetime
from strategies.debug_strategy import DebugStrategy
from strategies.always_buy_strategy import AlwaysBuyStrategy

HIST_DIR = "/root/falah-ai-bot/historical_data/"

def load_csv_file(path):
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.dropna(subset=['date', 'close', 'open', 'high', 'low', 'volume'])
        if len(df) < 100:
            print(f"⚠️ Skipping {os.path.basename(path)}, insufficient data ({len(df)} rows)")
            return None
        return df.sort_values('date')
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        return None

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(100000)

    strategy_choice = "debug"  # "debug" or "always"
    if strategy_choice == "debug":
        cerebro.addstrategy(DebugStrategy)
    else:
        cerebro.addstrategy(AlwaysBuyStrategy)

    csv_files = [os.path.join(HIST_DIR, f) for f in os.listdir(HIST_DIR) if f.endswith(".csv")]
    print(f"✅ Found {len(csv_files)} files.")

    valid = 0
    for f in csv_files:
        df = load_csv_file(f)
        if df is None:
            continue
        data = bt.feeds.PandasData(dataname=df, datetime='date', open='open', high='high',
                                   low='low', close='close', volume='volume', openinterest=None)
        cerebro.adddata(data)
        valid += 1

    print(f"✅ Loaded {valid} files.")

    if valid == 0:
        print("⚠️ No valid data found. Exiting.")
        exit()

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print("✅ Running backtest...")
    result = cerebro.run()[0]

    final_value = cerebro.broker.getvalue()
    print(f"🏁 Final Portfolio Value: ₹{final_value:.2f}")

    sharpe = result.analyzers.sharpe.get_analysis()
    print(f"📊 Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")

    trade_analysis = result.analyzers.trades.get_analysis()
    try:
        total_trades = trade_analysis.total.closed
    except (KeyError, AttributeError):
        total_trades = 0
    print(f"📈 Total Trades: {total_trades}")
