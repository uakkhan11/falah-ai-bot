# backtest_v2.py

import os, pandas as pd, numpy as np
from datetime import datetime

# --- CONFIGURATION ---
DATA_DIR        = "./historical_data"
SYMBOLS         = ["INFY","TCS","HDFCBANK"]
START_DATE      = "2018-01-01"
END_DATE        = "2023-12-31"
INITIAL_EQUITY  = 1_000_000
RISK_PER_TRADE  = 0.02    # 2% of current equity
COMMISSION      = 0.0005  # 0.05%
SLIPPAGE        = 0.0005  # 0.05%

# --- METRICS STORAGE ---
equity = INITIAL_EQUITY
peak   = INITIAL_EQUITY
drawdowns = []
equity_curve = []

trades = []  # list of dicts

# --- STRATEGY INTERFACE ---
class BaseStrategy:
    def generate_signals(self, df):
        """
        Should return a DataFrame with columns:
           'entry_price','stop_loss','target','exit_price' (nan if open)
        indexed by date.  Fill exit_price when SL/TP hit.
        """
        raise NotImplementedError

# --- EXAMPLE STRATEGIES ---
class EMACrossoverStrategy(BaseStrategy):
    def generate_signals(self, df):
        df = df.copy()
        df["EMA10"] = df["close"].ewm(span=10).mean()
        df["EMA21"] = df["close"].ewm(span=21).mean()
        signals = []
        in_trade = False

        for i in range(21, len(df)):
            date = df.index[i]
            price = df["close"].iat[i]
            if not in_trade and df["EMA10"].iat[i] > df["EMA21"].iat[i]:
                # enter
                sl = price * 0.98   # fixed 2% SL
                tp = price * 1.06   # fixed 3×SL ≈ 6%
                signals.append({"date": date, "entry": price, "sl": sl, "tp": tp})
                in_trade = True
            elif in_trade:
                # check SL/TP
                entry, sl, tp = (signals[-1]["entry"], signals[-1]["sl"], signals[-1]["tp"])
                low, high = df["low"].iat[i], df["high"].iat[i]
                if low <= sl:
                    exit_p = sl
                elif high >= tp:
                    exit_p = tp
                else:
                    continue
                signals[-1].update({"exit_date": date, "exit": exit_p})
                in_trade = False

        return pd.DataFrame(signals).set_index("date")

class RSIStrategy(BaseStrategy):
    def generate_signals(self, df):
        df = df.copy()
        df["RSI"] = pd.Series(df["close"]).rolling(14).apply(
            lambda x: np.where(
                (x.diff() > 0).sum() / 14 > 0.5, 1, 0
            )
        )
        # ... your RSI oversold logic ...
        return pd.DataFrame()  # implement similarly

# --- BACKTEST ENGINE ---
def backtest():
    global equity, peak

    strategies = [EMACrossoverStrategy() /*, RSIStrategy(), ...*/]

    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}.csv")
        if not os.path.exists(path):
            print(f"⚠️ Missing data for {sym}")
            continue

        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.loc[START_DATE:END_DATE]

        for strat in strategies:
            sigs = strat.generate_signals(df)
            for dt, row in sigs.iterrows():
                entry, sl, tp = row["entry"], row["sl"], row["tp"]
                # position sizing
                risk_amt = equity * RISK_PER_TRADE
                qty = int(risk_amt / (entry - sl))
                if qty <= 0: continue

                # simulate commission & slippage
                buy_price  = entry * (1 + SLIPPAGE)
                cost       = buy_price * qty * (1 + COMMISSION)

                exit_price = row.get("exit", np.nan)
                if pd.isna(exit_price):
                    # default to close price next bar
                    exit_price = df.loc[dt:]["close"].iat[1]
                sell_price = exit_price * (1 - SLIPPAGE)
                proceeds   = sell_price * qty * (1 - COMMISSION)

                pnl = proceeds - cost
                equity += pnl
                peak = max(peak, equity)
                dd   = (peak - equity) / peak
                drawdowns.append(dd)
                equity_curve.append({"date": dt, "equity": equity})

                trades.append({
                    "symbol": sym,
                    "strategy": strat.__class__.__name__,
                    "entry_date": dt,
                    "exit_date": row["exit_date"],
                    "qty": qty,
                    "entry": buy_price,
                    "exit": sell_price,
                    "pnl": pnl
                })

    # dump trades
    pd.DataFrame(trades).to_csv("all_trades.csv", index=False)
    # equity curve
    ec = pd.DataFrame(equity_curve).set_index("date")
    ec.to_csv("equity_curve.csv")
    # summary metrics
    ret = ec["equity"].pct_change().dropna()
    cagr = (equity / INITIAL_EQUITY) ** (1 / ((ec.index[-1]-ec.index[0]).days/365.)) -1
    sharpe = ret.mean()/ret.std()*np.sqrt(252)
    max_dd = max(drawdowns)*100

    print(f"CAGR {cagr:.2%} | Sharpe {sharpe:.2f} | MaxDD {max_dd:.1f}%")

if __name__=="__main__":
    backtest()
