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
        risk_per_trade=0.02,
        atr_multiplier=1.5,
        ai_threshold=0.6   # Relaxed threshold
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.ema10 = bt.indicators.EMA(self.data.close, period=self.p.ema_short)
        self.ema21 = bt.indicators.EMA(self.data.close, period=self.p.ema_long)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.order = None

    def next(self):
        if self.order:
            return  # Wait for pending order

        # Compute AI score safely
        vol_series = pd.Series(self.data.volume.get(size=10))
        if vol_series.isna().any() or vol_series.mean() == 0:
            print(f"{self.datetime.date()} {self.data._name} Skipping due to bad volume")
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

        # Entry Criteria
        ema_pass = self.ema10[0] > self.ema21[0]
        rsi_pass = self.rsi[0] > 45
        ai_pass = ai_score >= self.p.ai_threshold

        passed = sum([ema_pass, rsi_pass, ai_pass])
        entry_signal = passed >= 2

        print(
            f"{self.datetime.date()} {self.data._name} "
            f"EMA10:{self.ema10[0]:.2f} EMA21:{self.ema21[0]:.2f} "
            f"RSI:{self.rsi[0]:.2f} AI:{ai_score:.2f} "
            f"EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass} Passed:{passed}"
        )

        # Entry
        if not self.position and entry_signal:
            risk = self.p.risk_per_trade * self.broker.getvalue()
            sl = self.data.close[0] - self.p.atr_multiplier * self.atr[0]
            if sl >= self.data.close[0]:
                print("⚠️ Skipping trade: Stoploss >= Entry price")
                return

            qty = int(risk / (self.data.close[0] - sl))
            if qty <= 0:
                print("⚠️ Skipping trade: qty <=0")
                return

            self.order = self.buy(size=qty)
            self.sl_price = sl
            self.tp_price = self.data.close[0] + (self.data.close[0] - sl) * 3
            print(f"✅ Buy order placed: qty={qty} SL={self.sl_price:.2f} TP={self.tp_price:.2f}")

        # Exit logic
        if self.position:
            if self.data.low[0] <= self.sl_price:
                self.order = self.close()
                print(f"🛑 Stop Loss hit at {self.sl_price:.2f}")
            elif self.data.high[0] >= self.tp_price:
                self.order = self.close()
                print(f"✅ Target hit at {self.tp_price:.2f}")
