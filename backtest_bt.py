import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# ───── CONFIG ─────────────────────────────────────────────────
DATA_DIR       = "./historical_data"
INITIAL_EQUITY = 1_000_000
RISK_PER_TRADE = 0.02    # 2% of current equity
COMMISSION     = 0.0005  # 0.05%
SLIPPAGE       = 0.0005  # 0.05%

# ───── STRATEGY INTERFACE ─────────────────────────────────────
class BaseStrategy:
    def generate_signals(self, df):
        raise NotImplementedError

# ───── EXAMPLE STRATEGIES ─────────────────────────────────────
class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, span_short=10, span_long=21):
        self.span_short = span_short
        self.span_long = span_long

    def generate_signals(self, df):
        df = df.copy()
        df[f"EMA{self.span_short}"] = df["close"].ewm(span=self.span_short).mean()
        df[f"EMA{self.span_long}"]  = df["close"].ewm(span=self.span_long).mean()
        signals, in_trade = [], False
        for i in range(self.span_long, len(df)):
            date = df.index[i]
            price = df["close"].iat[i]
            ema_s = df[f"EMA{self.span_short}"].iat[i]
            ema_l = df[f"EMA{self.span_long}"].iat[i]
            if not in_trade and ema_s > ema_l:
                sl, tp = price * 0.98, price * 1.06
                signals.append({"entry_date": date, "entry": price, "sl": sl, "tp": tp})
                in_trade = True
            elif in_trade:
                low, high = df["low"].iat[i], df["high"].iat[i]
                exit_price = None
                if low <= signals[-1]["sl"]:
                    exit_price = signals[-1]["sl"]
                elif high >= signals[-1]["tp"]:
                    exit_price = signals[-1]["tp"]
                if exit_price is not None:
                    signals[-1].update({"exit_date": date, "exit": exit_price})
                    in_trade = False
        return signals

class RSIStrategy(BaseStrategy):
    def __init__(self, period=14, entry_thresh=30, exit_thresh=50):
        self.period = period
        self.entry = entry_thresh
        self.exit  = exit_thresh

    def generate_signals(self, df):
        df = df.copy()
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = -delta.clip(upper=0).rolling(self.period).mean()
        df["RSI"] = 100 - 100/(1 + gain/loss)
        signals, in_trade = [], False
        for i in range(self.period, len(df)):
            date, price, rsi = df.index[i], df["close"].iat[i], df["RSI"].iat[i]
            if not in_trade and rsi < self.entry:
                sl, tp = price * 0.98, price * 1.06
                signals.append({"entry_date": date, "entry": price, "sl": sl, "tp": tp})
                in_trade = True
            elif in_trade and rsi > self.exit:
                signals[-1].update({"exit_date": date, "exit": price})
                in_trade = False
        return signals

class VolumeBreakoutStrategy(BaseStrategy):
    def __init__(self, lookback=20, vol_multiplier=1.5):
        self.lookback = lookback
        self.mult     = vol_multiplier

    def generate_signals(self, df):
        df = df.copy()
        df["vol_avg"] = df["volume"].rolling(self.lookback).mean()
        signals, in_trade = [], False
        for i in range(self.lookback, len(df)):
            date = df.index[i]
            price = df["close"].iat[i]
            vol, avg = df["volume"].iat[i], df["vol_avg"].iat[i]
            if not in_trade and vol > self.mult * avg:
                sl, tp = price * 0.98, price * 1.06
                signals.append({"entry_date": date, "entry": price, "sl": sl, "tp": tp})
                in_trade = True
            elif in_trade:
                low, high = df["low"].iat[i], df["high"].iat[i]
                exit_price = None
                if low <= signals[-1]["sl"]:
                    exit_price = signals[-1]["sl"]
                elif high >= signals[-1]["tp"]:
                    exit_price = signals[-1]["tp"]
                if exit_price:
                    signals[-1].update({"exit_date": date, "exit": exit_price})
                    in_trade = False
        return signals

# Placeholder for AI based exit strategy if needed

# ───── BACKTEST ENGINE ────────────────────────────────────────
def backtest(symbols, start, end, strategies):
    equity       = INITIAL_EQUITY
    peak_equity  = equity
    drawdowns    = []
    equity_curve = []
    all_trades   = []

    for sym in symbols:
        path = os.path.join(DATA_DIR, f"{sym}.csv")
        if not os.path.exists(path):
            print(f"⚠️ Missing data for {sym}")
            continue
        df = pd.read_csv(path, parse_dates=["date"]) 
        df.set_index("date", inplace=True)
        df = df.loc[start:end]

        for strat in strategies:
            trades = strat.generate_signals(df)
            for t in trades:
                entry, sl, tp = t["entry"], t["sl"], t.get("tp")
                if entry <= sl: continue
                # position sizing
                risk_amount = equity * RISK_PER_TRADE
                qty = int(risk_amount / (entry - sl))
                if qty < 1: continue
                # simulate entry
                buy_price = entry * (1 + SLIPPAGE)
                cost = buy_price * qty * (1 + COMMISSION)
                # decide exit
                exit_price = t.get("exit")
                if exit_price is None:
                    idx = df.index.get_loc(t["entry_date"]) + 1
                    if idx < len(df): exit_price = df["close"].iat[idx]
                    else: continue
                sell_price = exit_price * (1 - SLIPPAGE)
                proceeds = sell_price * qty * (1 - COMMISSION)
                pnl = proceeds - cost
                equity += pnl
                peak_equity = max(peak_equity, equity)
                drawdowns.append((peak_equity - equity)/peak_equity)
                equity_curve.append({"date": t["exit_date"], "equity": equity})
                all_trades.append({
                    "symbol": sym,
                    "strategy": strat.__class__.__name__,
                    **t,
                    "qty": qty,
                    "pnl": pnl
                })

    # results
    if not equity_curve:
        print("⚠️ No equity data to compute metrics.")
        return
    ec = pd.DataFrame(equity_curve).set_index("date").sort_index()
    ec.to_csv("equity_curve.csv")
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv("all_trades.csv", index=False)

    returns = ec["equity"].pct_change().dropna()
    days = (ec.index[-1] - ec.index[0]).days / 365.25
    cagr = (equity / INITIAL_EQUITY)**(1/days) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns)>1 else np.nan
    max_dd = max(drawdowns)*100 if drawdowns else 0
    print(f"CAGR: {cagr:.2%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.1f}%")

# ───── CLI ENTRYPOINT ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run backtests on multiple strategies.")
    parser.add_argument("--symbols", nargs="+", default=["INFY"], help="List of symbols to backtest")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2023-12-31", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    strategies = [
        EMACrossoverStrategy(),
        RSIStrategy(),
        VolumeBreakoutStrategy(),
    ]
    backtest(args.symbols, args.start, args.end, strategies)
