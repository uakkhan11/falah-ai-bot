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
        atr_multiplier=2.0,
        ai_threshold=0.3,
        min_atr=0.05,
        exit_bars=6
    )

    def __init__(self):
        self.orders = dict()
        self.entry_bars = dict()
        self.trades_log = []
        self.equity_curve = []
        self.drawdowns = []
        self.peak = self.broker.getvalue()

        # Prepare indicators per data
        self.inds = {}
        for d in self.datas:
            self.inds[d] = dict(
                ema10=bt.ind.EMA(d.close, period=self.p.ema_short),
                ema21=bt.ind.EMA(d.close, period=self.p.ema_long),
                rsi=bt.ind.RSI(d.close, period=self.p.rsi_period),
                atr=bt.ind.ATR(d, period=self.p.atr_period)
            )

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt}: {txt}")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        pnl = trade.pnl
        dt = trade.data.datetime.datetime(0)
        self.trades_log.append({
            "date": dt,
            "symbol": trade.data._name,
            "pnl": pnl,
            "entry_price": trade.price,
            "size": trade.size
        })
        self.log(f"{trade.data._name}: 💰 Trade closed. P&L: ₹{pnl:.2f}")

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        value = self.broker.getvalue()
        self.equity_curve.append({"date": dt, "capital": value})

        self.peak = max(self.peak, value)
        dd = (self.peak - value) / self.peak
        self.drawdowns.append(dd)

        for d in self.datas:
            pos = self.getposition(d)
            inds = self.inds[d]

            # Skip if already have open order
            if d in self.orders and self.orders[d]:
                continue

            # Time-based exit
            if pos and (len(self) - self.entry_bars.get(d, 0)) >= self.p.exit_bars:
                self.close(data=d)
                self.log(f"{d._name}: ⏳ Time exit after {self.p.exit_bars} bars")
                continue

            # Skip if in position
            if pos:
                if d.low[0] <= self.sl_prices[d]:
                    self.close(data=d)
                    self.log(f"{d._name}: 🛑 Stop Loss hit at {self.sl_prices[d]:.2f}")
                elif d.high[0] >= self.tp_prices[d]:
                    self.close(data=d)
                    self.log(f"{d._name}: ✅ Target hit at {self.tp_prices[d]:.2f}")
                continue

            # ATR safety check
            if inds['atr'][0] <= 0 or pd.isna(inds['atr'][0]):
                continue

            # AI Features
            vol_ratio = 1.0
            features = [[
                inds['rsi'][0],
                inds['ema10'][0],
                inds['ema21'][0],
                inds['atr'][0],
                vol_ratio
            ]]
            prob = model.predict_proba(pd.DataFrame(
                features,
                columns=["RSI","EMA10","EMA21","ATR","VolumeChange"]
            ))[0][1]
            ai_score = prob * 5.0

            ema_pass = inds['ema10'][0] > inds['ema21'][0]
            rsi_pass = inds['rsi'][0] > 35
            entry_signal = ema_pass or rsi_pass

            self.log(
                f"{d._name}: EMA10:{inds['ema10'][0]:.2f} EMA21:{inds['ema21'][0]:.2f} "
                f"RSI:{inds['rsi'][0]:.2f} ATR:{inds['atr'][0]:.4f} "
                f"AI:{ai_score:.2f} Entry:{entry_signal}"
            )

            if entry_signal:
                risk = self.p.risk_per_trade * value
                sl = d.close[0] - self.p.atr_multiplier * inds['atr'][0]
                if sl >= d.close[0]:
                    continue

                qty = int(risk / max(d.close[0] - sl, 1e-6))
                if qty <= 0:
                    continue

                o = self.buy(data=d, size=qty)
                self.orders[d] = o
                self.entry_bars[d] = len(self)
                if not hasattr(self, "sl_prices"):
                    self.sl_prices = dict()
                    self.tp_prices = dict()
                self.sl_prices[d] = sl
                self.tp_prices[d] = d.close[0] + (d.close[0] - sl) * 2.0
                self.log(f"{d._name}: ✅ Buy order: qty={qty} SL={sl:.2f} TP={self.tp_prices[d]:.2f}")

    def stop(self):
        for d in self.datas:
            pos = self.getposition(d)
            if pos:
                dt = d.datetime.datetime(0)
                exit_price = d.close[0]
                pnl = (exit_price - pos.price) * pos.size
                self.close(data=d)
                self.trades_log.append({
                    "date": dt,
                    "symbol": d._name,
                    "pnl": pnl,
                    "entry_price": pos.price,
                    "size": pos.size
                })
                self.log(f"{d._name}: 🔚 Closing open position manually. Final P&L: ₹{pnl:.2f}")
