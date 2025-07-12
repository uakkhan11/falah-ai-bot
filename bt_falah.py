import backtrader as bt
import pandas as pd
import joblib

model = joblib.load("/root/falah-ai-bot/model.pkl")

trades = []
equity_curve = []
drawdowns = []

class FalahStrategy(bt.Strategy):
    params = dict(
        rsi_period=14,
        ema_short=10,
        ema_long=21,
        atr_period=14,
        risk_per_trade=0.005,
        atr_multiplier=1.5,
        ai_threshold=0.6,
        min_atr=0.3
    )

    def __init__(self):
        self.orders = [None] * len(self.datas)
        self.sl_prices = [None] * len(self.datas)
        self.tp_prices = [None] * len(self.datas)
        self.peaks = [self.broker.getvalue()] * len(self.datas)
        self.trades_log = []

        # Prepare indicators per data
        self.rsi = []
        self.ema10 = []
        self.ema21 = []
        self.atr = []
        for d in self.datas:
            self.rsi.append(bt.indicators.RSI(d.close, period=self.p.rsi_period))
            self.ema10.append(bt.indicators.EMA(d.close, period=self.p.ema_short))
            self.ema21.append(bt.indicators.EMA(d.close, period=self.p.ema_long))
            self.atr.append(bt.indicators.ATR(d, period=self.p.atr_period))

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f"{dt} {txt}")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        dt = self.datas[0].datetime.date(0)
        self.log(f"{trade.data._name} 💰 Closed P&L: ₹{trade.pnl:.2f}")
        self.trades_log.append({
            "date": dt,
            "symbol": trade.data._name,
            "pnl": trade.pnl,
            "entry_price": trade.price,
            "size": trade.size
        })

    def next(self):
        dt = self.datas[0].datetime.date(0)
        value = self.broker.getvalue()
        equity_curve.append({"date": dt, "capital": value})

        peak = max(self.peaks + [value])
        drawdowns.append((peak - value) / peak)

        for i, d in enumerate(self.datas):
            if self.orders[i]:
                continue

            # Skip if too low ATR
            if self.atr[i][0] < self.p.min_atr:
                continue

            # Compute AI
            vol_series = pd.Series(d.volume.get(size=10))
            if vol_series.isna().any() or vol_series.mean() == 0:
                continue
            vol_ratio = d.volume[0] / vol_series.mean()
            features = [[
                self.rsi[i][0],
                self.ema10[i][0],
                self.ema21[i][0],
                self.atr[i][0],
                vol_ratio
            ]]
            prob = model.predict_proba(pd.DataFrame(
                features,
                columns=["RSI","EMA10","EMA21","ATR","VolumeChange"]
            ))[0][1]
            ai_score = prob * 5.0

            ema_pass = self.ema10[i][0] > self.ema21[i][0]
            rsi_pass = self.rsi[i][0] > 45
            ai_pass = ai_score >= self.p.ai_threshold
            entry_signal = ai_pass and (ema_pass or rsi_pass)

            print(
                f"{d._name} EMA10:{self.ema10[i][0]:.2f} EMA21:{self.ema21[i][0]:.2f} "
                f"RSI:{self.rsi[i][0]:.2f} AI:{ai_score:.2f} Entry:{entry_signal} "
                f"EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass}"
            )

            if not self.getposition(d).size and entry_signal:
                risk = self.p.risk_per_trade * value
                sl = d.close[0] - self.p.atr_multiplier * self.atr[i][0]
                if sl >= d.close[0]:
                    continue
                qty = int(risk / (d.close[0] - sl))
                if qty <= 0:
                    continue
                self.orders[i] = self.buy(data=d, size=qty)
                self.sl_prices[i] = sl
                self.tp_prices[i] = d.close[0] + (d.close[0] - sl) * 3
                print(
                    f"{d._name} ✅ Buy: qty={qty} SL={self.sl_prices[i]:.2f} TP={self.tp_prices[i]:.2f}"
                )

            if self.getposition(d).size:
                if d.low[0] <= self.sl_prices[i]:
                    self.orders[i] = self.close(data=d)
                    print(f"{d._name} 🛑 SL hit at {self.sl_prices[i]:.2f}")
                elif d.high[0] >= self.tp_prices[i]:
                    self.orders[i] = self.close(data=d)
                    print(f"{d._name} ✅ TP hit at {self.tp_prices[i]:.2f}")

    def stop(self):
        for i, d in enumerate(self.datas):
            pos = self.getposition(d)
            if pos.size:
                exit_price = d.close[0]
                pnl = (exit_price - pos.price) * pos.size
                self.close(data=d)
                self.trades_log.append({
                    "date": d.datetime.date(0),
                    "symbol": d._name,
                    "pnl": pnl,
                    "entry_price": pos.price,
                    "size": pos.size
                })
                print(
                    f"{d._name} 🔚 Closing open position manually. Final P&L: ₹{pnl:.2f}"
                )
