import backtrader as bt
import pandas as pd
import random
import joblib

# Load AI model globally
MODEL = joblib.load("/root/falah-ai-bot/model.pkl")

class FalahStrategy(bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("ema_short", 10),
        ("ema_long", 21),
        ("atr_period", 14),
        ("risk_per_trade", 0.01),
        ("atr_multiplier", 2.0),
        ("ai_threshold", 0.3),
        ("min_atr", 0.05),
        ("exit_bars", 6),
    )

    def __init__(self):
        self.trades_log = []
        self.indicators = {}
        self.state = {}

        for d in self.datas:
            self.indicators[d._name] = {
                "rsi": bt.ind.RSI(d.close, period=self.p.rsi_period),
                "ema10": bt.ind.EMA(d.close, period=self.p.ema_short),
                "ema21": bt.ind.EMA(d.close, period=self.p.ema_long),
                "atr": bt.ind.ATR(d, period=self.p.atr_period),
            }
            self.state[d._name] = {
                "entry_bar": None,
                "buy_price": None,
                "sl_price": None,
                "tp_price": None,
            }

    def log(self, txt, dt=None):
        dt = dt or self.datetime.datetime(0)
        print(f"{dt} {txt}")

    def next(self):
        for d in self.datas:
            symbol = d._name
            ind = self.indicators[symbol]
            s = self.state[symbol]

            # Skip if indicators are not ready
            if any([
                pd.isna(ind["rsi"][0]),
                pd.isna(ind["ema10"][0]),
                pd.isna(ind["ema21"][0]),
                pd.isna(ind["atr"][0]),
            ]):
                continue

            close = d.close[0]
            rsi = ind["rsi"][0]
            ema10 = ind["ema10"][0]
            ema21 = ind["ema21"][0]
            atr = ind["atr"][0]
            dt = d.datetime.datetime(0)

            if atr < self.p.min_atr:
                continue

            # AI score simulation (replace this with real model if needed)
            ai_raw = random.uniform(0, 1)
            ai_score = round(ai_raw * 5, 2)

            ema_pass = ema10 > ema21
            rsi_pass = rsi > 50
            ai_pass = ai_score >= 1.0
            entry_signal = ema_pass and rsi_pass and ai_pass

            # 🔍 Full diagnostics
            print(
                f"{dt} {symbol}: EMA10:{ema10:.2f} EMA21:{ema21:.2f} RSI:{rsi:.2f} ATR:{atr:.4f} "
                f"AIraw:{ai_raw:.4f} AIscore:{ai_score:.2f} "
                f"Entry:{entry_signal} EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass}"
            )

            # Entry logic
            if entry_signal and not self.getposition(d).size:
                sl = close - self.p.atr_multiplier * atr
                tp = close + self.p.atr_multiplier * atr * 2
                value = self.broker.getvalue()
                risk_amount = self.p.risk_per_trade * value
                qty = int(risk_amount / (close - sl)) if (close - sl) > 0 else 0

                if qty <= 0:
                    self.log(f"{symbol}: ⛔ Skipped: qty<=0 or SL too close")
                    continue

                self.buy(data=d, size=qty)
                s["buy_price"] = close
                s["sl_price"] = sl
                s["tp_price"] = tp
                s["entry_bar"] = len(self)
                self.log(f"{symbol}: ✅ Buy order: qty={qty} SL={sl:.2f} TP={tp:.2f}")

            # Exit logic
            if self.getposition(d).size:
                if d.low[0] <= s["sl_price"]:
                    self.close(data=d)
                    self.log(f"{symbol}: 🛑 Stop Loss hit at {s['sl_price']:.2f}")
                elif d.high[0] >= s["tp_price"]:
                    self.close(data=d)
                    self.log(f"{symbol}: ✅ Target hit at {s['tp_price']:.2f}")
                elif len(self) - s["entry_bar"] >= self.p.exit_bars:
                    self.close(data=d)
                    self.log(f"{symbol}: ⏳ Time exit after {self.p.exit_bars} bars")

    def notify_trade(self, trade):
        if trade.isclosed:
            dt = self.datetime.datetime()
            pnl = trade.pnl
            symbol = trade.data._name
            self.trades_log.append({
                "symbol": symbol,
                "date": dt,
                "pnl": pnl,
                "entry_price": trade.price,
                "size": trade.size
            })
            self.log(f"{symbol}: 💰 Trade closed. P&L: ₹{pnl:.2f}")

    def stop(self):
        self.log("🔚 Strategy Finished")
