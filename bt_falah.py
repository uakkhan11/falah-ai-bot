# bt_falah.py

import backtrader as bt
import pandas as pd
import joblib
import random

model = joblib.load("/root/falah-ai-bot/model.pkl")

class FalahStrategy(bt.Strategy):
    def __init__(self):
        self.indicators = dict()

        for d in self.datas:
            self.indicators[d._name] = {
                'rsi': bt.ind.RSI(d.close, period=self.p.rsi_period),
                'ema10': bt.ind.EMA(d.close, period=self.p.ema_short),
                'ema21': bt.ind.EMA(d.close, period=self.p.ema_long),
                'atr': bt.ind.ATR(d, period=self.p.atr_period),
                'entry_bar': None,
                'order': None,
                'sl': None,
                'tp': None
            }

    def next(self):
        for d in self.datas:
            name = d._name
            close = d.close[0]
            dt = d.datetime.datetime(0).strftime('%Y-%m-%d %H:%M')

            # Skip if not enough candles
            if len(d) < 30:
                continue

            ind = self.indicators[name]
            rsi = ind['rsi'][0]
            ema10 = ind['ema10'][0]
            ema21 = ind['ema21'][0]
            atr = ind['atr'][0]

            # AI input
            vol_series = pd.Series(d.volume.get(size=10))
            vol_ratio = (vol_series.iloc[-1] / vol_series.mean()) if vol_series.mean() != 0 else 1.0
            features = pd.DataFrame([[
                rsi, ema10, ema21, atr, vol_ratio
            ]], columns=["RSI", "EMA10", "EMA21", "ATR", "VolumeChange"])
            ai_prob = model.predict_proba(features)[0][1]
            ai_score = round(ai_prob * 5.0, 2)

            # Entry condition
            ema_pass = ema10 > ema21
            rsi_pass = rsi > 50
            ai_pass = ai_score >= self.p.ai_threshold
            atr_pass = atr >= self.p.min_atr
            entry_signal = ema_pass and rsi_pass and ai_pass and atr_pass

            print(f"{dt} {name}: EMA10:{ema10:.2f} EMA21:{ema21:.2f} RSI:{rsi:.2f} ATR:{atr:.4f} "
                  f"AIscore:{ai_score:.2f} Entry:{entry_signal} "
                  f"EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass} ATRpass:{atr_pass}")

            # ENTRY
            if entry_signal and not self.getposition(d).size:
                value = self.broker.getvalue()
                risk = self.p.risk_per_trade * value
                sl = close - self.p.atr_multiplier * atr
                if sl >= close:
                    print(f"{dt} {name}: ⛔ Skipped: SL >= Entry ({sl:.2f} >= {close:.2f})")
                    continue

                qty = int(risk / (close - sl))
                if qty <= 0:
                    print(f"{dt} {name}: ⛔ Skipped: qty <= 0")
                    continue

                self.buy(data=d, size=qty)
                ind['sl'] = sl
                ind['tp'] = close + (close - sl) * 2
                ind['entry_bar'] = len(d)
                print(f"{dt} {name}: ✅ Buy order: qty={qty} SL={ind['sl']:.2f} TP={ind['tp']:.2f}")

            # EXIT
            if self.getposition(d).size:
                if d.low[0] <= ind['sl']:
                    print(f"{dt} {name}: 🛑 Stop Loss hit at {ind['sl']:.2f}")
                    self.close(data=d)
                elif d.high[0] >= ind['tp']:
                    print(f"{dt} {name}: ✅ Target hit at {ind['tp']:.2f}")
                    self.close(data=d)
                elif len(d) - ind['entry_bar'] >= self.p.exit_bars:
                    print(f"{dt} {name}: ⏳ Time exit after {self.p.exit_bars} bars")
                    self.close(data=d)
