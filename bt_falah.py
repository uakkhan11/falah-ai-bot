import backtrader as bt
import pandas as pd
import numpy as np
import joblib

MODEL = joblib.load("/root/falah-ai-bot/model.pkl")

class FalahStrategy(bt.Strategy):
    params = dict(
        rsi_period=14,
        ema_short=10,
        ema_long=21,
        atr_period=14,
        risk_per_trade=0.01,
        atr_multiplier=2.0,
        ai_threshold=2.0,
        min_atr=0.05,
        exit_bars=6,
        max_trades_per_day=5,
        max_daily_loss=10000
    )

    def __init__(self):
        self.order = None
        self.entry_bar = dict()
        self.buy_price = dict()
        self.sl_price = dict()
        self.tp_price = dict()
        self.trades_log = []
        self.daily_pnl = {}
        self.trade_count = {}

        self.indicators = {
            d._name: {
                "rsi": bt.ind.RSI(d.close, period=self.p.rsi_period),
                "ema10": bt.ind.EMA(d.close, period=self.p.ema_short),
                "ema21": bt.ind.EMA(d.close, period=self.p.ema_long),
                "atr": bt.ind.ATR(d, period=self.p.atr_period),
            } for d in self.datas
        }

    def log(self, txt, dt=None):
        dt = dt or self.datetime.datetime(0)
        print(f"{dt} {txt}")

    def next(self):
        for d in self.datas:
            symbol = d._name
            ind = self.indicators[symbol]

            # Skip if indicators are not ready
            if any(pd.isna(ind[key][0]) for key in ("rsi", "ema10", "ema21", "atr")):
                continue

            close = d.close[0]
            rsi = ind["rsi"][0]
            ema10 = ind["ema10"][0]
            ema21 = ind["ema21"][0]
            atr = ind["atr"][0]
            dt = d.datetime.datetime(0)
            today = d.datetime.date(0)

            # Volume Ratio
            try:
                vol_series = pd.Series(d.volume.get(size=10))
                if d.volume[0] <= 0 or vol_series[:-1].mean() == 0:
                    vol_ratio = 1.0
                else:
                    vol_ratio = vol_series[-1] / (vol_series[:-1].mean() + 1e-9)
            except Exception:
                vol_ratio = 1.0

            # Features for AI model
            features = [[rsi, ema10, ema21, atr, vol_ratio]]
            features_df = pd.DataFrame(features, columns=["RSI", "EMA10", "EMA21", "ATR", "VolumeChange"])

            if not np.isfinite(features_df.values).all():
                self.log(f"{symbol}: ❌ Skipping AI model — invalid input: {features_df.values}")
                continue

            # AI prediction
            try:
                ai_score = MODEL.predict_proba(features_df)[0][1] * 5  # scale to 0–5
                ai_raw = ai_score / 5.0
            except Exception as e:
                self.log(f"{symbol}: ❌ Model prediction failed: {e}")
                continue

            # Entry Conditions
            ema_pass = ema10 > ema21
            rsi_pass = rsi > 50
            ai_pass = ai_score >= 1.0
            entry_signal = ema_pass and rsi_pass and ai_pass

            print(
                f"{dt} {symbol}: EMA10:{ema10:.2f} EMA21:{ema21:.2f} RSI:{rsi:.2f} ATR:{atr:.4f} "
                f"AIraw:{ai_raw:.4f} AIscore:{ai_score:.2f} "
                f"Entry:{entry_signal} EMApass:{ema_pass} RSIpass:{rsi_pass} AIpass:{ai_pass}"
            )

            self.trade_count.setdefault(today, 0)
            self.daily_pnl.setdefault(today, 0)

            # Entry logic
            if entry_signal and not self.getposition(d).size:
                if self.trade_count[today] >= self.p.max_trades_per_day:
                    self.log(f"{symbol}: ❌ Max trades hit")
                    continue
                if self.daily_pnl[today] <= -self.p.max_daily_loss:
                    self.log(f"{symbol}: ❌ Max daily loss hit")
                    continue

                sl = close - self.p.atr_multiplier * atr
                tp = close + self.p.atr_multiplier * atr * 2
                risk = self.p.risk_per_trade * self.broker.getvalue()
                qty = int(risk / (close - sl)) if (close - sl) > 0 else 0

                if qty <= 0:
                    self.log(f"{symbol}: ⛔ Skipped: qty<=0 or SL too close")
                    continue

                self.buy(data=d, size=qty)
                self.buy_price[symbol] = close
                self.sl_price[symbol] = sl
                self.tp_price[symbol] = tp
                self.entry_bar[symbol] = len(self)
                self.trade_count[today] += 1
                self.log(f"{symbol}: ✅ Buy qty={qty} SL={sl:.2f} TP={tp:.2f}")

            # Exit logic
            if self.getposition(d).size:
                if d.low[0] <= self.sl_price[symbol]:
                    self.close(data=d)
                    self.log(f"{symbol}: 🛑 SL Hit at {self.sl_price[symbol]:.2f}")
                elif d.high[0] >= self.tp_price[symbol]:
                    self.close(data=d)
                    self.log(f"{symbol}: ✅ Target Hit at {self.tp_price[symbol]:.2f}")
                elif len(self) - self.entry_bar[symbol] >= self.p.exit_bars:
                    self.close(data=d)
                    self.log(f"{symbol}: ⏳ Timed exit after {self.p.exit_bars} bars")

    def notify_trade(self, trade):
        if trade.isclosed:
            dt = self.datetime.date()
            pnl = trade.pnl
            symbol = trade.data._name
            self.daily_pnl.setdefault(dt, 0)
            self.daily_pnl[dt] += pnl
            self.trades_log.append({
                "symbol": symbol,
                "date": dt,
                "pnl": pnl,
                "entry_price": trade.price,
                "size": trade.size
            })
            self.log(f"{symbol}: 💰 Closed P&L: ₹{pnl:.2f}")

    def stop(self):
        df = pd.DataFrame(self.trades_log)
        df.to_csv("backtest_trades.csv", index=False)
        self.log(f"🔚 Strategy Done. Trades saved to backtest_trades.csv")
