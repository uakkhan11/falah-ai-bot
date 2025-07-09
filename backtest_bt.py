import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from ai_engine import calculate_ai_exit_score

# ───── CONFIG ─────────────────────────────────────────────────
DATA_DIR       = './historical_data'
SYMBOLS        = ['INFY','TCS','HDFCBANK']
START_DATE     = datetime(2018,1,1)
END_DATE       = datetime(2023,12,31)
INITIAL_CASH   = 1_000_000
RISK_PER_TRADE = 0.02  # 2%
SLIPPAGE       = 0.0005
COMMISSION     = 0.0005
AI_THRESHOLD   = 70

# ───── STRATEGIES ───────────────────────────────────────────────
class EMACrossover(bt.Strategy):
    params = dict(ema1=10, ema2=21, risk=RISK_PER_TRADE)

    def __init__(self):
        self.ema1 = bt.ind.EMA(self.data.close, period=self.p.ema1)
        self.ema2 = bt.ind.EMA(self.data.close, period=self.p.ema2)
        self.order = None
        self.sl_price = None
        self.tp_price = None

    def next(self):
        if not self.position:
            if self.ema1[0] > self.ema2[0]:
                size = self.calc_size(self.data.close[0], self.data.low[-1]*0.98)
                self.sl_price = self.data.close[0]*0.98
                self.tp_price = self.data.close[0]*1.06
                self.order = self.buy(size=size)
        else:
            if self.data.close[0] <= self.sl_price or self.data.close[0] >= self.tp_price:
                self.close()

    def calc_size(self, price, stop_price):
        cash = self.broker.getcash()
        risk_amount = cash * self.p.risk
        return int(risk_amount / abs(price - stop_price))

class RSIStrategy(bt.Strategy):
    params = dict(period=14, lower=30, upper=50, risk=RISK_PER_TRADE)

    def __init__(self):
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.period)

    def next(self):
        if not self.position:
            if self.rsi[0] < self.p.lower:
                size = self.calc_size(self.data.close[0], self.data.close[0]*0.98)
                self.buy(size=size)
        else:
            if self.rsi[0] > self.p.upper:
                self.close()

    def calc_size(self, price, stop_price):
        cash = self.broker.getcash()
        risk_amount = cash * self.p.risk
        return int(risk_amount / abs(price - stop_price))

class VolumeBreakout(bt.Strategy):
    params = dict(period=20, mult=1.5, risk=RISK_PER_TRADE)

    def __init__(self):
        self.vol_avg = bt.ind.SimpleMovingAverage(self.data.volume, period=self.p.period)

    def next(self):
        if not self.position:
            if self.data.volume[0] > self.vol_avg[0] * self.p.mult:
                size = self.calc_size(self.data.close[0], self.data.close[0]*0.98)
                self.buy(size=size)
        else:
            if self.data.close[0] <= self.data.close[0]*0.98:
                self.close()

    def calc_size(self, price, stop_price):
        cash = self.broker.getcash()
        risk_amount = cash * self.p.risk
        return int(risk_amount / abs(price - stop_price))

class AIScoreExit(bt.Strategy):
    params = dict(threshold=AI_THRESHOLD, risk=RISK_PER_TRADE)

    def __init__(self):
        self.sma = bt.ind.SimpleMovingAverage(self.data.close, period=20)
        self.hist = []

    def next(self):
        self.hist.append({
            'date': self.data.datetime.datetime(0),
            'open': self.data.open[0], 'high': self.data.high[0],
            'low': self.data.low[0], 'close': self.data.close[0],
            'volume': self.data.volume[0]
        })
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                atr = np.mean([h['high']-h['low'] for h in self.hist[-14:]])
                stop = self.data.close[0] - 1.5*atr
                size = self.calc_size(self.data.close[0], stop)
                self.buy(size=size)
                self.sl = stop
        else:
            df = pd.DataFrame(self.hist)
            ai_score, _ = calculate_ai_exit_score(df, self.sl, self.data.close[0], atr_value=None)
            if ai_score >= self.p.threshold:
                self.close()

    def calc_size(self, price, stop_price):
        cash = self.broker.getcash()
        risk_amount = cash * self.p.risk
        return int(risk_amount / abs(price - stop_price))

# ───── RUN BACKTEST ────────────────────────────────────────────
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(perc=SLIPPAGE)

    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f'{sym}.csv')
        data = bt.feeds.GenericCSVData(
            dataname=path,
            dtformat='%Y-%m-%d',
            datetime=0, open=1, high=2, low=3, close=4, volume=5,
            fromdate=START_DATE, todate=END_DATE,
            timeframe=bt.TimeFrame.Days
        )
        cerebro.adddata(data, name=sym)

    # Add all strategies
    cerebro.addstrategy(EMACrossover)
    cerebro.addstrategy(RSIStrategy)
    cerebro.addstrategy(VolumeBreakout)
    cerebro.addstrategy(AIScoreExit)

    print('Starting Portfolio Value:', cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value:', cerebro.broker.getvalue())
    cerebro.plot()
