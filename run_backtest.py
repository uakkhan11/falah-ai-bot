import backtrader as bt
import pandas as pd
from strategies.debug_strategy import DebugStrategy

cerebro = bt.Cerebro()

data = pd.read_csv('data.csv')  # your historical OHLCV data
data['datetime'] = pd.to_datetime(data['date'])
data.set_index('datetime', inplace=True)

feed = bt.feeds.PandasData(dataname=data)

cerebro.adddata(feed)
cerebro.addstrategy(DebugStrategy)
cerebro.broker.set_cash(100000)
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

results = cerebro.run()
strat = results[0]

sharpe = strat.analyzers.sharpe.get_analysis()
trade_analysis = strat.analyzers.trades.get_analysis()

print(f"\n🏁 Final Portfolio Value: ₹{cerebro.broker.getvalue():.2f}")
print(f"📊 Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
total_trades = trade_analysis.total.closed if 'total' in trade_analysis and hasattr(trade_analysis.total, 'closed') else 0
print(f"📈 Total Trades: {total_trades}")
