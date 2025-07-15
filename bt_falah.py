import backtrader as bt
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

MODEL = joblib.load("/root/falah-ai-bot/model.pkl")


class FalahBacktest(bt.Strategy):
    params = dict(
        ai_threshold=1.0,
        min_atr=0.05,
        atr_multiplier=2.0,
        exit_bars=6,
        risk_per_trade=0.01,
        trailing_sl_pct=0.5,
        rsi_period=14,
        ema_short=10,
        ema_long=21,
        atr_period=14,
    )

    def __init__(self):
        self.order = None
        self.sl_price = {}
        self.tp_price = {}
        self.entry_bar = {}
        self.trades_log = []
        self.last_ai_score = {}

        self.indicators = {}
        for d in self.datas:
            self.indicators[d._name] = dict(
                rsi=bt.ind.RSI(d.close, period=self.p.rsi_period),
                ema10=bt.ind.EMA(d.close, period=self.p.ema_short),
                ema21=bt.ind.EMA(d.close, period=self.p.ema_long),
                atr=bt.ind.ATR(d, period=self.p.atr_period),
            )

    def log(self, txt, dt=None):
        dt = dt or self.datetime.datetime(0)
        print(f"{dt.isoformat()} {txt}")

    def next(self):
        for d in self.datas:
            name = d._name
            ind = self.indicators[name]

            if any(np.isnan(ind[key][0]) for key in ind):
                continue

            close = d.close[0]
            rsi = ind["rsi"][0]
            ema10 = ind["ema10"][0]
            ema21 = ind["ema21"][0]
            atr = ind["atr"][0]

            if atr < self.p.min_atr:
                continue

            try:
                vol = np.array([d.volume[-i] for i in range(1, 11)])
                volume_change = vol[-1] / (np.mean(vol[:-1]) + 1e-9)
            except:
                volume_change = 1.0

            features = pd.DataFrame([{
                "RSI": rsi,
                "EMA10": ema10,
                "EMA21": ema21,
                "ATR": atr,
                "VolumeChange": volume_change
            }])

            if not np.isfinite(features.values).all():
                continue

            ai_score = MODEL.predict_proba(features)[0][1] * 5
            self.last_ai_score[name] = ai_score

            ema_pass = ema10 > ema21
            rsi_pass = rsi > 50
            ai_pass = ai_score >= self.p.ai_threshold

            position = self.getposition(d).size

            if not position and ema_pass and rsi_pass and ai_pass:
                sl = close - self.p.atr_multiplier * atr
                tp = close + self.p.atr_multiplier * atr * 2

                cash = self.broker.get_cash()
                risk = cash * self.p.risk_per_trade
                qty = int(risk / (close - sl)) if (close - sl) > 0 else 0

                if qty > 0:
                    self.order = self.buy(data=d, size=qty)
                    self.sl_price[name] = sl
                    self.tp_price[name] = tp
                    self.entry_bar[name] = len(self)
                    self.log(f"{name}: BUY {qty} @ {close:.2f}, SL={sl:.2f}, TP={tp:.2f}, AI={ai_score:.2f}")

            if position:
                if d.low[0] <= self.sl_price[name]:
                    self.close(data=d)
                    self.log(f"{name}: SL Hit at {close:.2f}")
                elif d.high[0] >= self.tp_price[name]:
                    self.close(data=d)
                    self.log(f"{name}: TP Hit at {close:.2f}")
                elif len(self) - self.entry_bar[name] >= self.p.exit_bars:
                    self.close(data=d)
                    self.log(f"{name}: Timed Exit at {close:.2f}")

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"{trade.data._name}: Closed PnL: {trade.pnl:.2f}")
            self.trades_log.append({
                "symbol": trade.data._name,
                "pnl": trade.pnl,
                "ai_score": self.last_ai_score.get(trade.data._name, None),
            })

    def stop(self):
        if not os.path.exists("./backtest_results"):
            os.makedirs("./backtest_results")
        pd.DataFrame(self.trades_log).to_csv("./backtest_results/backtest_trades.csv", index=False)
        self.log(f"🔚 Backtest complete. Trades saved.")


def run_backtest():
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1000000)

    files = [f for f in os.listdir("./data/") if f.endswith(".csv")]
    print(f"✅ Found {len(files)} CSV files.")

    for f in files:
        df = pd.read_csv(f"./data/{f}")
        if "date" not in df.columns:
            print(f"Skipping {f}, missing 'date' column.")
            continue

        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna()

        data = bt.feeds.PandasData(dataname=df, datetime="date", open="open", high="high",
                                   low="low", close="close", volume="volume", openinterest=None)
        cerebro.adddata(data, name=f.replace(".csv", ""))

    cerebro.addstrategy(FalahBacktest)
    results = cerebro.run()
    cerebro.broker.getvalue()
    cerebro.plot()


if __name__ == "__main__":
    run_backtest()
