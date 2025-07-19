import backtrader as bt
import pandas as pd
import os
import glob
from datetime import datetime

class DebugStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def next(self):
        if self.rsi[0] < 30:
            self.log(f'BUY CREATE {self.data.close[0]:.2f}, RSI={self.rsi[0]:.2f}')
            self.buy()
        elif self.rsi[0] > 70:
            self.log(f'SELL CREATE {self.data.close[0]:.2f}, RSI={self.rsi[0]:.2f}')
            self.sell()

cerebro = bt.Cerebro()
cerebro.broker.set_cash(100000)

symbols = []
total_trades = 0
symbol_win_rate = {}

for filepath in glob.glob('data/*.csv'):
    symbol = os.path.basename(filepath).replace('.csv', '')
    print(f'\n=== Running backtest for {symbol} ===')
    try:
        df = pd.read_csv(filepath)
        if len(df) < 100:
            print(f'Skipping {symbol}, insufficient data.')
            continue

        df['datetime'] = pd.to_datetime(df['date'])
        df.set_index('datetime', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]

        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data, name=symbol)

    except Exception as e:
        print(f'Skipping {symbol}, error: {e}')
        continue
    symbols.append(symbol)

cerebro.addstrategy(DebugStrategy)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")

print(f"\nTotal symbols loaded: {len(symbols)}")
results = cerebro.run()

trade_analysis = results[0].analyzers.trade_analyzer.get_analysis()
total_trades = trade_analysis.total.closed if 'total' in trade_analysis and 'closed' in trade_analysis.total else 0

print("\n===== FINAL SUMMARY =====")
print(f"Total Symbols Backtested: {len(symbols)}")
print(f"Total Trades Executed: {total_trades}")
if 'won' in trade_analysis and 'total' in trade_analysis.won and total_trades:
    win_ratio = (trade_analysis.won.total / total_trades) * 100
    print(f"Win Ratio: {win_ratio:.2f}%")
else:
    print("Win Ratio: N/A")

print("\n===== END =====")
