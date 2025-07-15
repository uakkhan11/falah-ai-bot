import backtrader as bt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

MODEL = joblib.load("/root/falah-ai-bot/model.pkl")

class FalahStrategy(bt.Strategy):
    params = dict(
        rsi_period=14,
        ema_short=10,
        ema_long=21,
        atr_period=14,
        risk_per_trade=0.01,
        atr_multiplier=2.0,
        ai_threshold=1.0,
        min_atr=0.05,
        exit_bars=6,
        max_trades_per_day=5,
        max_daily_loss=10000,
        use_trailing_sl=True,
        trailing_sl_pct=0.5
    )

    def __init__(self):
        self.order = {}
        self.entry_bar = {}
        self.buy_price = {}
        self.sl_price = {}
        self.tp_price = {}
        self.trades_log = []
        self.daily_pnl = {}
        self.trade_count = {}
        self.last_ai_score = {}
        self.last_features = {}

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
    dt = self.datetime.date()
    if dt not in self.daily_pnl:
        self.daily_pnl[dt] = 0
        self.trade_count[dt] = 0

    for d in self.datas:
        symbol = d._name
        ind = self.indicators[symbol]

        # Skip NaN indicator values
        if any(pd.isna(ind[key][0]) for key in ("rsi", "ema10", "ema21", "atr")):
            self.log(f"{symbol}: Skipped due to NaN indicators")
            continue

        close = d.close[0]
        rsi = ind["rsi"][0]
        ema10 = ind["ema10"][0]
        ema21 = ind["ema21"][0]
        atr = ind["atr"][0]

        if atr < self.p.min_atr:
            self.log(f"{symbol}: Skipped due to ATR {atr:.2f} < {self.p.min_atr}")
            continue

        try:
            vol_series = pd.Series(d.volume.get(size=10))
            vol_mean = vol_series[:-1].mean()
            vol_ratio = vol_series[-1] / (vol_mean + 1e-9) if vol_mean > 0 else 1.0
        except Exception:
            vol_ratio = 1.0

        features_df = pd.DataFrame([{
            "RSI": rsi,
            "EMA10": ema10,
            "EMA21": ema21,
            "ATR": atr,
            "VolumeChange": vol_ratio
        }])

        if not np.isfinite(features_df.values).all():
            self.log(f"{symbol}: Skipped due to non-finite features")
            continue

        try:
            prob = MODEL.predict_proba(features_df)[0][1]
            ai_score = prob * 5
        except Exception as e:
            self.log(f"{symbol}: Skipped due to AI model error: {e}")
            continue

        self.last_ai_score[symbol] = ai_score
        self.last_features[symbol] = features_df.iloc[0].to_dict()

        ema_pass = ema10 > ema21
        rsi_pass = rsi > 50
        ai_pass = ai_score >= self.p.ai_threshold

        self.log(f"{symbol}: close={close:.2f}, RSI={rsi:.2f}, EMA10={ema10:.2f}, EMA21={ema21:.2f}, ATR={atr:.2f}, VolChange={vol_ratio:.2f}, AI Prob={prob:.2f}, AI Score={ai_score:.2f}")

        entry_signal = ema_pass and rsi_pass and ai_pass
        pos = self.getposition(d).size

        if entry_signal and not pos:
            if self.trade_count[dt] >= self.p.max_trades_per_day:
                self.log(f"{symbol}: Skipped due to max trades per day")
                continue
            if self.daily_pnl[dt] <= -self.p.max_daily_loss:
                self.log(f"{symbol}: Skipped due to daily loss limit")
                continue

            sl = close - self.p.atr_multiplier * atr
            risk = self.p.risk_per_trade * self.broker.getvalue()
            qty = int(risk / (close - sl)) if (close - sl) > 0 else 0

            if qty <= 0:
                self.log(f"{symbol}: Skipped due to invalid qty calc (SL too close)")
                continue

            self.buy(data=d, size=qty)
            self.buy_price[symbol] = close
            self.sl_price[symbol] = sl
            self.entry_bar[symbol] = len(self)
            self.trade_count[dt] += 1
            self.log(f"{symbol}: ✅ BUY {qty} at {close:.2f}, SL {sl:.2f}, AI Score {ai_score:.2f}")

            if pos:
                reason = None

                if d.low[0] <= self.sl_price.get(symbol, 0):
                    reason = "SL Hit"
                elif d.high[0] >= self.tp_price.get(symbol, 0):
                    reason = "TP Hit"
                elif len(self) - self.entry_bar.get(symbol, 0) >= self.p.exit_bars:
                    reason = "Timed Exit"
                elif self.p.use_trailing_sl:
                    new_sl = close - self.p.trailing_sl_pct * atr
                    if new_sl > self.sl_price.get(symbol, 0):
                        self.sl_price[symbol] = new_sl

                if reason:
                    self.close(data=d)
                    self.log(f"{symbol}: 🔁 {reason} at {close:.2f}")

    def notify_trade(self, trade):
        if trade.isclosed:
            dt = self.datetime.date()
            symbol = trade.data._name
            pnl = trade.pnl
            self.daily_pnl[dt] += pnl

            log = {
                "symbol": symbol,
                "date": dt,
                "pnl": pnl,
                "entry_price": trade.price,
                "exit_price": trade.price + pnl / trade.size if trade.size != 0 else None,
                "size": trade.size,
                "ai_score": self.last_ai_score.get(symbol),
                **self.last_features.get(symbol, {})
            }
            self.trades_log.append(log)
            self.log(f"{symbol}: 💰 Closed P&L: ₹{pnl:.2f}")

    def stop(self):
        df = pd.DataFrame(self.trades_log)
        out_path = "./backtest_results/backtest_trades.csv"
        df.to_csv(out_path, index=False)
        self.log(f"🔚 Strategy done. Trades saved to {out_path}")
