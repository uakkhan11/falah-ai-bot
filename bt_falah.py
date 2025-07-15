import backtrader as bt
import pandas as pd
import numpy as np
import joblib
import os

MODEL = joblib.load("/root/falah-ai-bot/model.pkl")

class FalahDebugStrategy(bt.Strategy):
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
        self.trades_log = []
        self.daily_pnl = {}
        self.trade_count = {}
        self.last_ai_score = {}
        self.last_features = {}

        # Debug counters
        self.total_symbols = len(self.datas)
        self.valid_indicator_count = 0
        self.ai_score_count = 0
        self.executed_trades_count = 0

        self.indicators = {
            d._name: {
                "rsi": bt.ind.RSI(d.close, period=self.p.rsi_period),
                "ema10": bt.ind.EMA(d.close, period=self.p.ema_short),
                "ema21": bt.ind.EMA(d.close, period=self.p.ema_long),
                "atr": bt.ind.ATR(d, period=self.p.atr_period),
            } for d in self.datas
        }

    def log(self, txt, dt=None):
        dt = dt or self.datetime.date(0)
        print(f"{dt} {txt}")

    def next(self):
        dt = self.datetime.date()
        if dt not in self.daily_pnl:
            self.daily_pnl[dt] = 0
            self.trade_count[dt] = 0

        for d in self.datas:
            symbol = d._name
            ind = self.indicators[symbol]

            if any(pd.isna(ind[key][0]) for key in ("rsi", "ema10", "ema21", "atr")):
                continue

            self.valid_indicator_count += 1

            close = d.close[0]
            rsi = ind["rsi"][0]
            ema10 = ind["ema10"][0]
            ema21 = ind["ema21"][0]
            atr = ind["atr"][0]

            if atr < self.p.min_atr:
                continue

            try:
                vol_series = pd.Series(d.volume.get(size=10))
                vol_ratio = vol_series[-1] / (vol_series[:-1].mean() + 1e-9) if vol_series[:-1].mean() > 0 else 1.0
            except Exception:
                vol_ratio = 1.0

            features = [[rsi, ema10, ema21, atr, vol_ratio]]
            features_df = pd.DataFrame(features, columns=["RSI", "EMA10", "EMA21", "ATR", "VolumeChange"])

            if not np.isfinite(features_df.values).all():
                continue

            try:
                ai_score = MODEL.predict_proba(features_df)[0][1] * 5
            except Exception:
                continue

            self.last_ai_score[symbol] = ai_score
            self.last_features[symbol] = features_df.iloc[0].to_dict()

            if ai_score >= self.p.ai_threshold:
                self.ai_score_count += 1

            ema_pass = ema10 > ema21
            rsi_pass = rsi > 50
            ai_pass = ai_score >= self.p.ai_threshold
            entry_signal = ema_pass and rsi_pass and ai_pass

            pos = self.getposition(d).size

            if entry_signal and not pos:
                sl = close - self.p.atr_multiplier * atr
                qty = 1  # Minimal qty for backtest tracking
                self.buy(data=d, size=qty)
                self.executed_trades_count += 1
                self.log(f"{symbol}: ✅ Buy at {close:.2f} AI={ai_score:.2f}")

    def stop(self):
        print("\n🔎 Backtest Summary")
        print(f"Total Symbols Loaded: {self.total_symbols}")
        print(f"Symbols with Valid Indicators: {self.valid_indicator_count}")
        print(f"Symbols Passed AI Score Threshold: {self.ai_score_count}")
        print(f"Total Trades Executed: {self.executed_trades_count}")
        print(f"Total Trades Logged: {len(self.trades_log)}")
        print("✅ Backtest completed.\n")
