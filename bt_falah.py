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
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.ema10 = bt.indicators.EMA(self.data.close, period=self.p.ema_short)
        self.ema21 = bt.indicators.EMA(self.data.close, period=self.p.ema_long)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.order = None
        self.entry_bar = None
        self.trades_log = []
        self.equity_curve = []
        self.drawdowns = []
        self.peak = self.broker.getvalue()

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
        dt = self.datas[0].datetime.datetime(0).strftime("%Y-%m-%d %H:%M")
        symbol = self.datas[0]._name
        close = self.datas[0].close[0]
        ema10_val = self.ema10[0]
        ema21_val = self.ema21[0]
        rsi_val = self.rsi[0]
        atr_val = self.atr[0]

        # Simulated AI score
    ai_raw = random.uniform(0, 1)
    ai_score = round(ai_raw * 5, 2)  # Scale to 0–5


    # Entry conditions
    ema_pass = ema10_val > ema21_val
    rsi_pass = rsi_val > 50
    ai_pass = ai_score >= 1.0
    entry_signal = ema_pass and rsi_pass and ai_pass

    # 🖨️ Print diagnostics for each symbol
    print(f"{dt} {symbol}: EMA10:{ema10_val:.2f} EMA21:{ema21_val:.2f} RSI:{rsi_val:.2f} "
          f"ATR:{atr_val:.2f} AIraw:{ai_raw:.4f} AIscore:{ai_score:.2f} "
          f"Entry:{entry_signal} EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass}")
    
    # Only act if entry signal passes
    if entry_signal and not self.position:
        sl = close * 0.98
        tp = close * 1.04
        qty = 100  # placeholder
        print(f"{dt} {symbol}: ✅ Buy order: qty={qty} SL={sl:.2f} TP={tp:.2f}")
        self.buy_price = close
        self.buy(size=qty)
        
        if self.order:
            return

        vol_series = pd.Series(self.data.volume.get(size=10))
        vol_ratio = 1.0

        features = [[
            self.rsi[0],
            self.ema10[0],
            self.ema21[0],
            self.atr[0],
            vol_ratio
        ]]
        prob = model.predict_proba(pd.DataFrame(
            features,
            columns=["RSI", "EMA10", "EMA21", "ATR", "VolumeChange"]
        ))[0][1]
        ai_score = prob * 5.0

        ema_pass = self.ema10[0] > self.ema21[0]
        rsi_pass = self.rsi[0] > 35
        ai_pass = ai_score >= self.p.ai_threshold
        entry_signal = ema_pass or rsi_pass

        self.log(
            f"EMA10:{self.ema10[0]:.2f} EMA21:{self.ema21[0]:.2f} "
            f"RSI:{self.rsi[0]:.2f} ATR:{self.atr[0]:.4f} "
            f"AIraw:{prob:.4f} AIscore:{ai_score:.2f} "
            f"Entry:{entry_signal} EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass}"
        )

        if not self.position and entry_signal:
            risk = self.p.risk_per_trade * value
            sl = self.data.close[0] - self.p.atr_multiplier * self.atr[0]
            if sl >= self.data.close[0]:
                self.log(f"⛔ Skipped: SL >= Entry price ({sl:.2f} >= {self.data.close[0]:.2f})")
                return
            qty = int(risk / (self.data.close[0] - sl))
            if qty <= 0:
                self.log("⛔ Skipped: qty <=0")
                return
            self.order = self.buy(size=qty)
            self.sl_price = sl
            self.tp_price = self.data.close[0] + (self.data.close[0] - sl) * 2.0
            self.entry_bar = len(self)
            self.log(f"✅ Buy order: qty={qty} SL={self.sl_price:.2f} TP={self.tp_price:.2f}")

        if self.position:
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
