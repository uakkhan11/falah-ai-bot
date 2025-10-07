#!/usr/bin/env python3
import os
import argparse
import glob
import math
import pandas as pd
import numpy as np
from datetime import datetime

# Import paths from improved_fetcher if available; else set manually
try:
    from improved_fetcher import BASE_DIR, DATA_DIRS
except Exception:
    BASE_DIR = "/root/falah-ai-bot"
    DATA_DIRS = {"daily": os.path.join(BASE_DIR, "swing_data")}

DAILY_DIR = DATA_DIRS["daily"]
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_symbol_df(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Ensure required columns
    for col in ["open","high","low","close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")
    # Fill indicators if missing
    if "atr" not in df.columns or df["atr"].isna().all():
        tr1 = (df["high"] - df["low"])
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14, min_periods=14).mean()
    if "adx" not in df.columns or df["adx"].isna().all():
        up = df["high"].diff()
        down = -df["low"].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr1 = (df["high"] - df["low"])
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean().replace(0, np.nan)
        plus_di = 100 * pd.Series(plus_dm).rolling(14, min_periods=14).sum() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14, min_periods=14).sum() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        df["adx"] = dx.rolling(14, min_periods=14).mean()
    # Compute EMAs for strategy
    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df

def backtest_symbol(df, symbol, capital, risk_per_trade, atr_mult, trail):
    """
    Next-day execution on daily signals:
    - Entry if on day t (prior close) ema5 crosses above ema20 AND close>ema200 AND adx>=25
    - Initial stop = close_t - atr_mult * atr_t
    - Qty sized by risk_per_trade * equity / (close_t - stop)
    - Exit on cross-down on prior bar confirmation or stop breach; optional ATR trailing
    """
    trades = []
    in_pos = False
    qty = 0
    entry = 0.0
    stop = 0.0
    trail_stop = None
    equity = capital

    # Use indexes to avoid lookahead: evaluate signal at t-1, trade at t
    for i in range(201, len(df)-1):
        # Prior bar for signal
        ema5_prev2 = df.loc[i-1-1, "ema5"]
        ema20_prev2 = df.loc[i-1-1, "ema20"]
        ema5_prev  = df.loc[i-1, "ema5"]
        ema20_prev = df.loc[i-1, "ema20"]

        crossed_up = (ema5_prev2 <= ema20_prev2) and (ema5_prev > ema20_prev)
        crossed_down = (ema5_prev2 >= ema20_prev2) and (ema5_prev < ema20_prev)
        c_prev = df.loc[i-1, "close"]
        atr_prev = df.loc[i-1, "atr"]
        adx_prev = df.loc[i-1, "adx"]
        ema200_prev = df.loc[i-1, "ema200"]

        # Execution day = i (next session open or close proxy); here use open of i
        exec_open = df.loc[i, "open"]
        exec_close = df.loc[i, "close"]
        date_entry = df.loc[i, "date"]

        if not in_pos:
            if crossed_up and c_prev > ema200_prev and (adx_prev if not np.isnan(adx_prev) else 0) >= 25 and not np.isnan(atr_prev):
                sl = c_prev - atr_mult * atr_prev
                risk_per_share = max(0.01, c_prev - sl)
                max_risk = risk_per_trade * equity
                qty = int(max(1, math.floor(max_risk / risk_per_share)))
                if qty > 0:
                    in_pos = True
                    entry = exec_open
                    stop = entry - atr_mult * atr_prev  # anchor initial stop to entry
                    trail_stop = stop
                    trades.append({
                        "symbol": symbol,
                        "side": "BUY",
                        "date": date_entry,
                        "price": entry,
                        "qty": qty
                    })
        else:
            # Update trailing stop with latest ATR based on prior close
            if trail and not np.isnan(atr_prev):
                candidate = exec_close - atr_mult * atr_prev
                trail_stop = max(trail_stop, candidate)

            # Check stop breach intraday using low of exec day
            low_i = df.loc[i, "low"]
            exit_reason = None
            exit_price = None

            if low_i <= trail_stop:
                exit_reason = "STOP"
                # Assume worst-case: stop executes near stop level
                exit_price = trail_stop
            elif crossed_down:
                exit_reason = "XDOWN"
                exit_price = exec_open

            if exit_reason:
                pnl = (exit_price - entry) * qty
                equity += pnl
                trades.append({
                    "symbol": symbol,
                    "side": "SELL",
                    "date": df.loc[i, "date"],
                    "price": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "reason": exit_reason
                })
                in_pos = False
                qty = 0
                entry = 0.0
                stop = 0.0
                trail_stop = None

    return trades

def aggregate_reports(trades, out_trades_csv, out_summary_csv):
    if len(trades) == 0:
        pd.DataFrame([], columns=["symbol","side","date","price","qty","pnl","reason"]).to_csv(out_trades_csv, index=False)
        pd.DataFrame([{"metric":"trades","value":0}]).to_csv(out_summary_csv, index=False)
        return

    tdf = pd.DataFrame(trades)
    tdf["date"] = pd.to_datetime(tdf["date"])
    # Build round-trip trade pairs for metrics
    closed = []
    pos = {}
    for _, r in tdf.iterrows():
        key = r["symbol"]
        if r["side"] == "BUY":
            pos[key] = r
        else:
            if key in pos:
                buy = pos.pop(key)
                sell = r
                rr = {
                    "symbol": key,
                    "entry_date": buy["date"],
                    "entry_price": buy["price"],
                    "exit_date": sell["date"],
                    "exit_price": sell["price"],
                    "qty": buy["qty"],
                    "pnl": sell.get("pnl", (sell["price"]-buy["price"])*buy["qty"]),
                    "reason": sell.get("reason","EXIT")
                }
                closed.append(rr)
    cdf = pd.DataFrame(closed)
    if cdf.empty:
        cdf = pd.DataFrame(columns=["symbol","entry_date","entry_price","exit_date","exit_price","qty","pnl","reason"])

    # Portfolio metrics
    total_pnl = cdf["pnl"].sum() if not cdf.empty else 0.0
    wins = (cdf["pnl"] > 0).sum() if not cdf.empty else 0
    losses = (cdf["pnl"] <= 0).sum() if not cdf.empty else 0
    win_rate = (wins / max(1, len(cdf))) * 100
    avg_win = cdf.loc[cdf["pnl"] > 0, "pnl"].mean() if not cdf.empty else 0.0
    avg_loss = cdf.loc[cdf["pnl"] <= 0, "pnl"].mean() if not cdf.empty else 0.0
    pf = (cdf.loc[cdf["pnl"] > 0, "pnl"].sum() / abs(cdf.loc[cdf["pnl"] <= 0, "pnl"].sum())) if (not cdf.empty and (cdf["pnl"] <= 0).any()) else np.nan

    # Simple equity and drawdown curve by trade close
    cdf = cdf.sort_values("exit_date").reset_index(drop=True)
    cdf["equity"] = cdf["pnl"].cumsum()
    peak = cdf["equity"].cummax() if not cdf.empty else pd.Series(dtype=float)
    dd = peak - cdf["equity"] if not cdf.empty else pd.Series(dtype=float)
    max_dd = dd.max() if not dd.empty else 0.0

    summary = pd.DataFrame([{
        "trades": int(len(cdf)),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 3) if not np.isnan(pf) else None,
        "total_pnl": round(total_pnl, 2),
        "max_drawdown": round(float(max_dd), 2),
        "avg_win": round(avg_win, 2) if not np.isnan(avg_win) else None,
        "avg_loss": round(avg_loss, 2) if not np.isnan(avg_loss) else None
    }])

    # Save
    tdf.to_csv(out_trades_csv, index=False)
    summary.to_csv(out_summary_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200000)
    ap.add_argument("--risk_per_trade", type=float, default=0.01, help="fraction of equity risked per trade (e.g., 0.01)")
    ap.add_argument("--atr_mult", type=float, default=2.0)
    ap.add_argument("--trail", type=str, default="true", help="true/false for ATR trailing")
    ap.add_argument("--limit_symbols", type=int, default=0, help="debug: test first N symbols")
    args = ap.parse_args()

    symbols = []
    files = sorted(glob.glob(os.path.join(DAILY_DIR, "*.csv")))
    if args.limit_symbols > 0:
        files = files[:args.limit_symbols]

    all_trades = []
    for f in files:
        symbol = os.path.basename(f).replace(".csv","")
        try:
            df = load_symbol_df(f)
        except Exception as e:
            continue
        trades = backtest_symbol(
            df, symbol,
            capital=args.capital,
            risk_per_trade=args.risk_per_trade,
            atr_mult=args.atr_mult,
            trail=(args.trail.lower()=="true")
        )
        all_trades.extend(trades)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_trades = os.path.join(REPORTS_DIR, f"ema_daily_trades_{ts}.csv")
    out_summary = os.path.join(REPORTS_DIR, f"ema_daily_summary_{ts}.csv")
    aggregate_reports(all_trades, out_trades, out_summary)
    print(f"Saved trades: {out_trades}")
    print(f"Saved summary: {out_summary}")

if __name__ == "__main__":
    main()
