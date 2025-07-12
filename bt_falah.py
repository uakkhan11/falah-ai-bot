import backtrader as bt
import pandas as pd

class FalahSanityStrategy(bt.Strategy):
    params = dict(
        risk_per_trade=0.01,
        atr_multiplier=2.0,
        exit_bars=6
    )

    def __init__(self):
        self.order = None
        self.entry_bar = None
        self.trades_log = []
        self.equity_curve = []
        self.drawdowns = []
        self.peak = self.broker.getvalue()
        self.atr = bt.indicators.ATR(self.data, period=14)

    def log(self, txt):
        dt = self.data.datetime.datetime(0)
        print(f"{dt} {self.data._name}: {txt}")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        pnl = trade.pnl
        dt = self.data.datetime.datetime(0)
        self.log(f"💰 Trade closed. P&L: ₹{pnl:.2f}")
        self.trades_log.append({
            "date": dt,
            "symbol": self.data._name,
            "pnl": pnl,
            "entry_price": trade.price,
            "size": trade.size
        })

    def next(self):
        dt = self.data.datetime.datetime(0)
        value = self.broker.getvalue()
        self.equity_curve.append({"date": dt, "capital": value})

        self.peak = max(self.peak, value)
        dd = (self.peak - value) / self.peak
        self.drawdowns.append(dd)

        if self.order:
            return

        # Sanity check: always enter if not in position
        if not self.position:
            risk = self.p.risk_per_trade * value
            sl = self.data.close[0] - self.p.atr_multiplier * self.atr[0]
            if sl >= self.data.close[0]:
                sl = self.data.close[0] - 0.5  # Force a stoploss

            qty = max(1, int(risk / (self.data.close[0] - sl)))
            self.order = self.buy(size=qty)
            self.sl_price = sl
            self.tp_price = self.data.close[0] + (self.data.close[0] - sl) * 2.0
            self.entry_bar = len(self)
            self.log(f"✅ Buy order: qty={qty} SL={self.sl_price:.2f} TP={self.tp_price:.2f}")

        else:
            if self.data.low[0] <= self.sl_price:
                self.order = self.close()
                self.log(f"🛑 Stop Loss hit at {self.sl_price:.2f}")
            elif self.data.high[0] >= self.tp_price:
                self.order = self.close()
                self.log(f"✅ Target hit at {self.tp_price:.2f}")
            elif len(self) - self.entry_bar >= self.p.exit_bars:
                self.order = self.close()
                self.log(f"⏳ Time exit after {self.p.exit_bars} bars")

    def stop(self):
        if self.position:
            dt = self.data.datetime.datetime(0)
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
