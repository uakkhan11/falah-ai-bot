import backtrader as bt
import pandas as pd
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
        today = self.datetime.date(0)
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0
            self.trade_count[today] = 0

        for d in self.datas:
            symbol = d._name
            ind = self.indicators[symbol]

            # Skip if indicators not ready
            if any(pd.isna(i[0]) for i in ind.values()):
                continue

            close = d.close[0]
            rsi = ind["rsi"][0]
            ema10 = ind["ema10"][0]
            ema21 = ind["ema21"][0]
            atr = ind["atr"][0]
            vol = d.volume[0]
            dt = d.datetime.datetime(0)

            vol_series = pd.Series(d.volume.get(size=10))
            vol_change = vol_series.pct_change().mean() if len(vol_series) > 1 else 0

            # AI prediction
            features = pd.DataFrame([[rsi, ema10, ema21, atr, vol_change]],
                                    columns=["RSI", "EMA10", "EMA21", "ATR", "VolumeChange"])
            ai_score = MODEL.predict_proba(features)[0][1] * 5

            ema_pass = ema10 > ema21
            rsi_pass = rsi > 55
            ai_pass = ai_score >= self.p.ai_threshold
            entry_signal = ema_pass and rsi_pass and ai_pass

            # 🖨️ Diagnostic output
            print(f"{dt} {symbol}: EMA10:{ema10:.2f} EMA21:{ema21:.2f} RSI:{rsi:.2f} ATR:{atr:.4f} "
                  f"AIscore:{ai_score:.2f} Entry:{entry_signal} TradesToday:{self.trade_count[today]} "
                  f"LossToday:₹{self.daily_pnl[today]:.2f}")

            # ✅ Entry logic
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
                    continue

                self.buy(data=d, size=qty)
                self.buy_price[symbol] = close
                self.sl_price[symbol] = sl
                self.tp_price[symbol] = tp
                self.entry_bar[symbol] = len(self)
                self.trade_count[today] += 1
                self.log(f"{symbol}: ✅ Buy qty={qty} SL={sl:.2f} TP={tp:.2f}")

            # Exit Logic
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
        # Export logs to CSV
        df = pd.DataFrame(self.trades_log)
        df.to_csv("backtest_trades.csv", index=False)
        self.log(f"🔚 Strategy Done. Trades saved to backtest_trades.csv")
