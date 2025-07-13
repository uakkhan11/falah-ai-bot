import backtrader as bt
import pandas as pd
import random
import joblib

# Load model once globally (optional)
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
        self.order = None
        self.entry_bar = None
        self.buy_price = None
        self.sl_price = None
        self.tp_price = None
        self.trades_log = []

        # Initialize indicators for each data feed
        self.indicators = dict()
        for d in self.datas:
            self.indicators[d._name] = {
                "rsi": bt.ind.RSI(d.close, period=self.p.rsi_period),
                "ema10": bt.ind.EMA(d.close, period=self.p.ema_short),
                "ema21": bt.ind.EMA(d.close, period=self.p.ema_long),
                "atr": bt.ind.ATR(d, period=self.p.atr_period),
            }

    def log(self, txt, dt=None):
        dt = dt or self.datetime.datetime(0)
        print(f"{dt} {txt}")

    def next(self):
        for d in self.datas:
            symbol = d._name
            if not d._maystart:  # skip if not enough bars
                continue

            close = d.close[0]
            rsi = self.indicators[symbol]["rsi"][0]
            ema10 = self.indicators[symbol]["ema10"][0]
            ema21 = self.indicators[symbol]["ema21"][0]
            atr = self.indicators[symbol]["atr"][0]
            dt = d.datetime.datetime(0)

            if pd.isna(close) or pd.isna(rsi) or pd.isna(ema10) or pd.isna(ema21) or pd.isna(atr):
                continue  # skip incomplete data

            # Simulated AI score or use model (you can replace this logic)
            ai_raw = random.uniform(0, 1)
            ai_score = round(ai_raw * 5, 2)  # Scale to 0–5

            # Entry Conditions
            ema_pass = ema10 > ema21
            rsi_pass = rsi > 50
            ai_pass = ai_score >= 1.0
            entry_signal = ema_pass and rsi_pass and ai_pass

            # 🔍 Print status for every symbol and bar
            print(
                f"{dt} {symbol}: EMA10:{ema10:.2f} EMA21:{ema21:.2f} RSI:{rsi:.2f} ATR:{atr:.4f} "
                f"AIraw:{ai_raw:.4f} AIscore:{ai_score:.2f} "
                f"Entry:{entry_signal} EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass}"
            )

            # Execute Buy
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
                self.buy_price = close
                self.sl_price = sl
                self.tp_price = tp
                self.entry_bar = len(self)
                self.log(f"{symbol}: ✅ Buy order: qty={qty} SL={sl:.2f} TP={tp:.2f}")

            # Manage open position
            if self.getposition(d).size:
                if d.low[0] <= self.sl_price:
                    self.close(data=d)
                    self.log(f"{symbol}: 🛑 Stop Loss hit at {self.sl_price:.2f}")
                elif d.high[0] >= self.tp_price:
                    self.close(data=d)
                    self.log(f"{symbol}: ✅ Target hit at {self.tp_price:.2f}")
                elif len(self) - self.entry_bar >= self.p.exit_bars:
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
