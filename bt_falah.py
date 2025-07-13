# bt_falah.py

import backtrader as bt
import pandas as pd
import joblib
import random

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
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        self.ema10 = bt.ind.EMA(self.data.close, period=self.p.ema_short)
        self.ema21 = bt.ind.EMA(self.data.close, period=self.p.ema_long)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)

        self.order = None
        self.entry_bar = None
        self.sl_price = None
        self.tp_price = None
        self.trades_log = []

    def log(self, txt):
        dt = self.data.datetime.datetime(0)
        print(f"{dt} {self.data._name}: {txt}")

    def notify_trade(self, trade):
        if trade.isclosed:
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
        if not self.data.close or len(self.data) < 30:
            return

        dt = self.data.datetime.datetime(0).strftime("%Y-%m-%d %H:%M")
        symbol = self.data._name
        close = self.data.close[0]
        ema10_val = self.ema10[0]
        ema21_val = self.ema21[0]
        rsi_val = self.rsi[0]
        atr_val = self.atr[0]

        # Prepare input features for AI model
        vol_series = pd.Series(self.data.volume.get(size=10))
        vol_ratio = (vol_series.iloc[-1] / vol_series.mean()) if vol_series.mean() != 0 else 1.0
        features = pd.DataFrame([[
            rsi_val,
            ema10_val,
            ema21_val,
            atr_val,
            vol_ratio
        ]], columns=["RSI", "EMA10", "EMA21", "ATR", "VolumeChange"])

        ai_prob = model.predict_proba(features)[0][1]
        ai_score = round(ai_prob * 5.0, 2)

        # Entry conditions
        ema_pass = ema10_val > ema21_val
        rsi_pass = rsi_val > 50
        ai_pass = ai_score >= self.p.ai_threshold
        entry_signal = ema_pass and rsi_pass and ai_pass and atr_val >= self.p.min_atr

        self.log(f"EMA10:{ema10_val:.2f} EMA21:{ema21_val:.2f} RSI:{rsi_val:.2f} ATR:{atr_val:.4f} "
                 f"AIraw:{ai_prob:.4f} AIscore:{ai_score:.2f} Entry:{entry_signal} "
                 f"EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass}")

        # ENTRY
        if entry_signal and not self.position:
            value = self.broker.getvalue()
            risk_amount = self.p.risk_per_trade * value
            sl = close - self.p.atr_multiplier * atr_val
            if sl >= close:
                self.log(f"⛔ Skipped: SL ≥ Entry ({sl:.2f} ≥ {close:.2f})")
                return
            qty = int(risk_amount / (close - sl))
            if qty <= 0:
                self.log("⛔ Skipped: qty <= 0")
                return

            self.order = self.buy(size=qty)
            self.sl_price = sl
            self.tp_price = close + (close - sl) * 2
            self.entry_bar = len(self)
            self.log(f"✅ Buy order: qty={qty} SL={self.sl_price:.2f} TP={self.tp_price:.2f}")

        # EXIT
        if self.position:
            if self.data.low[0] <= self.sl_price:
                self.log(f"🛑 Stop Loss hit at {self.sl_price:.2f}")
                self.close()
            elif self.data.high[0] >= self.tp_price:
                self.log(f"✅ Target hit at {self.tp_price:.2f}")
                self.close()
            elif len(self) - self.entry_bar >= self.p.exit_bars:
                self.log(f"⏳ Time exit after {self.p.exit_bars} bars")
                self.close()

    def stop(self):
        # Final cleanup
        if self.position:
            dt = self.data.datetime.datetime(0)
            exit_price = self.data.close[0]
            pnl = (exit_price - self.position.price) * self.position.size
            self.close()
            self.log(f"🔚 Closing open position manually. Final P&L: ₹{pnl:.2f}")
