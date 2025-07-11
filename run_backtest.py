import backtrader as bt
from bt_falah import FalahStrategy

cerebro = bt.Cerebro()
cerebro.broker.setcash(1_000_000)
cerebro.broker.setcommission(commission=0.0005)

data = bt.feeds.GenericCSVData(
    dataname="/root/falah-ai-bot/historical_data/NIFTY.csv",
    dtformat="%Y-%m-%d %H:%M:%S%z",
    timeframe=bt.TimeFrame.Days,
    compression=1,
    openinterest=-1
)

cerebro.adddata(data)
cerebro.addstrategy(FalahStrategy)
print("Starting Portfolio Value:", cerebro.broker.getvalue())
cerebro.run()
print("Ending Portfolio Value:", cerebro.broker.getvalue())
#cerebro.plot()
