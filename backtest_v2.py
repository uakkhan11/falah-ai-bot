import os
import pandas as pd
import numpy as np
from datetime import datetime

# ───── CONFIG ─────────────────────────────────────────────────
DATA_DIR        = "./historical_data"
SYMBOLS         = ["INFY","TCS","HDFCBANK"]
START_DATE      = "2018-01-01"
END_DATE        = "2023-12-31"
INITIAL_EQUITY  = 1_000_000
RISK_PER_TRADE  = 0.02    # 2% of current equity
COMMISSION      = 0.0005  # 0.05%
SLIPPAGE        = 0.0005  # 0.05%

# ───── STORAGE ────────────────────────────────────────────────
equity        = INITIAL_EQUITY
peak_equity   = INITIAL_EQUITY
drawdowns     = []
equity_curve  = []
all_trades    = []

# ───── STRATEGY INTERFACE ─────────────────────────────────────
class BaseStrategy:
    def generate_signals(self, df):
        raise NotImplementedError

# (Your strategy classes go here...)

def backtest():
    global equity, peak_equity

    # instantiate your strategies
    strategies = [
        EMACrossoverStrategy(),
        RSIStrategy(),
        VolumeBreakoutStrategy(),
        AIScoreExitStrategy(ai_exit_threshold=70)
    ]

    for sym in SYMBOLS:
        path = os.path.join(DATA_DIR, f"{sym}.csv")
        if not os.path.exists(path):
            print(f"⚠️ Missing data for {sym}")
            continue

        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.loc[START_DATE:END_DATE]

        for strat in strategies:
            trades = strat.generate_signals(df)
            for t in trades:
                entry_date = t["entry_date"]
                exit_date  = t.get("exit_date")
                entry, sl, tp = t["entry"], t["sl"], t["tp"]

                # position sizing
                risk_amount = equity * RISK_PER_TRADE
                qty = int(risk_amount / (entry - sl))
                if qty < 1:
                    continue

                # simulate entry
                buy_price = entry * (1 + SLIPPAGE)
                cost = buy_price * qty * (1 + COMMISSION)

                # decide exit
                if not exit_date or t.get("exit") is None:
                    # fallback to next day's close
                    idx = df.index.get_loc(entry_date)
                    if idx + 1 < len(df):
                        exit_price = df["close"].iat[idx+1]
                        exit_date = df.index[idx+1]
                    else:
                        continue
                else:
                    exit_price = t["exit"]

                # simulate exit
                sell_price = exit_price * (1 - SLIPPAGE)
                proceeds = sell_price * qty * (1 - COMMISSION)

                # pnl & equity update
                pnl = proceeds - cost
                equity += pnl
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity
                drawdowns.append(dd)

                # record equity
                equity_curve.append({
                    "date": exit_date,
                    "equity": equity
                })

                # record trade
                all_trades.append({
                    "symbol": sym,
                    "strategy": strat.__class__.__name__,
                    "entry_date": entry_date,
                    "exit_date":  exit_date,
                    "qty":        qty,
                    "entry":      buy_price,
                    "exit":       sell_price,
                    "pnl":        pnl
                })

    # → save trades
    if all_trades:
        pd.DataFrame(all_trades).to_csv("all_trades.csv", index=False)
    else:
        print("⚠️ No trades generated.")

    # → equity curve & performance
    if equity_curve:
        ec = pd.DataFrame(equity_curve).set_index("date").sort_index()
        ec.to_csv("equity_curve.csv")

        returns = ec["equity"].pct_change().dropna()
        days = (ec.index[-1] - ec.index[0]).days / 365.25
        cagr = (equity / INITIAL_EQUITY) ** (1 / days) - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan
        max_dd = max(drawdowns) * 100 if drawdowns else 0.0

        print(f"CAGR: {cagr:.2%}  |  Sharpe: {sharpe:.2f}  |  Max DD: {max_dd:.1f}%")
    else:
        print("⚠️ No equity data to compute performance metrics.")

if __name__ == "__main__":
    backtest()
