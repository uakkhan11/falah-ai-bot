import backtrader as bt
import pandas as pd
import joblib

model = joblib.load("/root/falah-ai-bot/model.pkl")

class FalahStrategy(bt.Strategy):
    params = dict(
        rsi_period=14,
        ema_short=10,
        ema_long=21,
        atr_period=14,
        risk_per_trade=0.01,
        atr_multiplier=1.5,
        ai_threshold=0.4,
        min_atr=0.1
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.ema10 = bt.indicators.EMA(self.data.close, period=self.p.ema_short)
        self.ema21 = bt.indicators.EMA(self.data.close, period=self.p.ema_long)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.order = None
        self.peak = self.broker.getvalue()

        # Initialize logs
        self.trades_log = []
        self.equity_curve = []
        self.drawdowns = []

    def log(self, txt):
        dt = self.data.datetime.date(0)
        print(f"{dt} {self.data._name}: {txt}")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        pnl = trade.pnl
        dt = self.data.datetime.date(0)
        self.log(f"💰 Trade closed. P&L: ₹{pnl:.2f}")

        self.trades_log.append({
            "date": dt,
            "symbol": self.data._name,
            "pnl": pnl,
            "entry_price": trade.price,
            "size": trade.size
        })

    def next(self):
        if self.order:
            return

        dt = self.data.datetime.date(0)
        value = self.broker.getvalue()
        self.equity_curve.append({"date": dt, "capital": value})

        self.peak = max(self.peak, value)
        dd = (self.peak - value) / self.peak
        self.drawdowns.append(dd)

        # ATR check
        if self.atr[0] < self.p.min_atr:
            return

        # AI features
        vol_series = pd.Series(self.data.volume.get(size=10))
        if vol_series.isna().any() or vol_series.mean() == 0:
            return

        vol_ratio = self.data.volume[0] / vol_series.mean()
        features = [[
            self.rsi[0],
            self.ema10[0],
            self.ema21[0],
            self.atr[0],
            vol_ratio
        ]]
        prob = model.predict_proba(pd.DataFrame(
            features,
            columns=["RSI","EMA10","EMA21","ATR","VolumeChange"]
        ))[0][1]
        ai_score = prob * 5.0

        ema_pass = self.ema10[0] > self.ema21[0]
        rsi_pass = self.rsi[0] > 40
        ai_pass = ai_score >= self.p.ai_threshold

        entry_signal = ai_pass or ema_pass or rsi_pass

        self.log(
            f"EMA10:{self.ema10[0]:.2f} EMA21:{self.ema21[0]:.2f} "
            f"RSI:{self.rsi[0]:.2f} AI:{ai_score:.2f} Entry:{entry_signal} "
            f"EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass}"
        )

        if not self.position and entry_signal:
            risk = self.p.risk_per_trade * value
            sl = self.data.close[0] - self.p.atr_multiplier * self.atr[0]
            if sl >= self.data.close[0]:
                return

            qty = int(risk / (self.data.close[0] - sl))
            if qty <= 0:
                return

            self.order = self.buy(size=qty)
            self.sl_price = sl
            self.tp_price = self.data.close[0] + (self.data.close[0] - sl) * 2.5
            self.log(f"✅ Buy order: qty={qty} SL={self.sl_price:.2f} TP={self.tp_price:.2f}")

        if self.position:
            if self.data.low[0] <= self.sl_price:
                self.order = self.close()
                self.log(f"🛑 Stop Loss hit at {self.sl_price:.2f}")
            elif self.data.high[0] >= self.tp_price:
                self.order = self.close()
                self.log(f"✅ Target hit at {self.tp_price:.2f}")

    def stop(self):
        if self.position:
            dt = self.data.datetime.date(0)
            exit_price = self.data.close[0]
            pnl = (exit_price - self.position.price) * self.position.size

            self.close()
            self.trades_log.append({
                "date": dt,
                "symbol": self.data._name,
                "pnl": pnl,
                "entry_price": self.position.price,
                "size": self.position.size
            })

            self.log(f"🔚 Closing open position manually. Final P&L: ₹{pnl:.2f}")
