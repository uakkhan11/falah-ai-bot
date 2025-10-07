#!/usr/bin/env python3
import os
import argparse
import glob
import math
import pandas as pd
import numpy as np
from datetime import datetime

# Try to reuse paths from improved_fetcher.py
try:
    from improved_fetcher import BASE_DIR, DATA_DIRS
except Exception:
    BASE_DIR = "/root/falah-ai-bot"
    DATA_DIRS = {"daily": os.path.join(BASE_DIR, "swing_data")}

DAILY_DIR = DATA_DIRS["daily"]
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---------------- Data loading and indicators ----------------

def load_symbol_df(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Sanity columns
    for col in ["open","high","low","close"]:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")

    # ATR14
    if "atr" not in df.columns or df["atr"].isna().all():
        tr1 = (df["high"] - df["low"])
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14, min_periods=14).mean()

    # ADX14
    if "adx" not in df.columns or df["adx"].isna().all():
        up = df["high"].diff()
        down = -df["low"].diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0))
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0))
        tr1 = (df["high"] - df["low"])
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean().replace(0, np.nan)
        plus_di = 100 * plus_dm.rolling(14, min_periods=14).sum() / atr
        minus_di = 100 * minus_dm.rolling(14, min_periods=14).sum() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        df["adx"] = dx.rolling(14, min_periods=14).mean()

    # EMAs
    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df

# ---------------- Strategy engine ----------------

def backtest_symbol(df, symbol, capital, risk_per_trade, atr_mult, adx_threshold, trail):
    """
    Prior-bar confirmation (t-1), execute at next day's open (t), stop breach evaluated on day t low.
    """
    trades = []
    in_pos = False
    qty = 0
    entry = 0.0
    trail_stop = None
    equity = capital

    # need >= 200 bars for ema200 warm-up; iterate leaving 1 bar for execution
    for i in range(201, len(df)-1):
        # Signal day: i-1
        ema5_prev2 = df.loc[i-2, "ema5"]
        ema20_prev2 = df.loc[i-2, "ema20"]
        ema5_prev  = df.loc[i-1, "ema5"]
        ema20_prev = df.loc[i-1, "ema20"]
        crossed_up = (ema5_prev2 <= ema20_prev2) and (ema5_prev > ema20_prev)
        crossed_down = (ema5_prev2 >= ema20_prev2) and (ema5_prev < ema20_prev)

        c_prev = df.loc[i-1, "close"]
        atr_prev = df.loc[i-1, "atr"]
        adx_prev = df.loc[i-1, "adx"] if not np.isnan(df.loc[i-1, "adx"]) else 0.0
        ema200_prev = df.loc[i-1, "ema200"]

        # Execution day: i
        exec_open = df.loc[i, "open"]
        exec_close = df.loc[i, "close"]
        low_i = df.loc[i, "low"]
        date_i = df.loc[i, "date"]

        if not in_pos:
            if crossed_up and c_prev > ema200_prev and adx_prev >= adx_threshold and not np.isnan(atr_prev):
                # Risk-aware sizing
                sl_from_signal = c_prev - atr_mult * atr_prev
                risk_per_share = max(0.01, c_prev - sl_from_signal)
                max_risk = risk_per_trade * equity
                qty = int(max(1, math.floor(max_risk / risk_per_share)))
                if qty > 0:
                    in_pos = True
                    entry = exec_open
                    # anchor stop to entry
                    trail_stop = entry - atr_mult * atr_prev
                    trades.append({"symbol":symbol,"side":"BUY","date":date_i,"price":entry,"qty":qty})
        else:
            # update trailing stop based on prior bar ATR and latest close
            if trail and not np.isnan(atr_prev):
                candidate = exec_close - atr_mult * atr_prev
                trail_stop = max(trail_stop, candidate)

            exit_reason = None
            exit_price = None
            if low_i <= trail_stop:
                exit_reason = "STOP"
                exit_price = trail_stop
            elif crossed_down:
                exit_reason = "XDOWN"
                exit_price = exec_open

            if exit_reason:
                pnl = (exit_price - entry) * qty
                equity += pnl
                trades.append({"symbol":symbol,"side":"SELL","date":date_i,"price":exit_price,"qty":qty,"pnl":pnl,"reason":exit_reason})
                in_pos = False
                qty = 0
                entry = 0.0
                trail_stop = None

    return trades

# ---------------- Analytics ----------------

def build_daily_equity(trades, start_capital=0.0):
    """
    Build a unique daily equity curve even if multiple exits occur on the same date
    across many symbols. We first aggregate P&L by calendar date, then reindex.
    """
    if len(trades) == 0:
        return pd.DataFrame(columns=["date","equity","ret","drawdown"])

    tdf = pd.DataFrame(trades)
    tdf["date"] = pd.to_datetime(tdf["date"])
    # Consider only realized P&L on SELL legs
    exits = tdf[(tdf["side"]=="SELL") & (tdf["pnl"].notna())].copy()
    if exits.empty:
        # No realized exits yet; flat equity series with a single point
        return pd.DataFrame([{"date": pd.Timestamp.today().normalize(), "equity": start_capital, "ret": 0.0, "drawdown": 0.0}])

    # Aggregate P&L by calendar date to ensure unique index
    daily_pnl = exits.groupby(exits["date"].dt.normalize())["pnl"].sum().to_frame("pnl").sort_index()

    # Build cumulative equity
    daily_pnl["equity"] = start_capital + daily_pnl["pnl"].cumsum()

    # Create a continuous daily index and forward-fill equity
    full_idx = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="D")
    daily = daily_pnl.reindex(full_idx)
    daily["equity"] = daily["equity"].ffill().fillna(start_capital)

    # Compute daily returns and drawdowns
    daily["ret"] = daily["equity"].pct_change().fillna(0.0)
    daily["peak"] = daily["equity"].cummax()
    daily["drawdown"] = daily["peak"] - daily["equity"]

    daily = daily.reset_index().rename(columns={"index":"date"})
    return daily[["date","equity","ret","drawdown"]]


def summarize(trades, daily, start_capital):
    tdf = pd.DataFrame(trades)
    tdf["date"] = pd.to_datetime(tdf["date"])
    # round-trips
    pos = {}
    closed = []
    for _, r in tdf.sort_values(["symbol","date"]).iterrows():
        k = r["symbol"]
        if r["side"]=="BUY":
            pos[k] = r
        elif r["side"]=="SELL" and k in pos:
            b = pos.pop(k)
            c = {
                "symbol": k,
                "entry_date": b["date"],
                "entry_price": b["price"],
                "exit_date": r["date"],
                "exit_price": r["price"],
                "qty": b["qty"],
                "pnl": r.get("pnl", (r["price"]-b["price"])*b["qty"]),
                "reason": r.get("reason","EXIT")
            }
            closed.append(c)
    cdf = pd.DataFrame(closed).sort_values("exit_date").reset_index(drop=True)

    total_trades = len(cdf)
    wins = int((cdf["pnl"] > 0).sum()) if total_trades else 0
    losses = int((cdf["pnl"] <= 0).sum()) if total_trades else 0
    gross_profit = float(cdf.loc[cdf["pnl"] > 0, "pnl"].sum()) if total_trades else 0.0
    gross_loss = float(cdf.loc[cdf["pnl"] <= 0, "pnl"].sum()) if total_trades else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else np.nan
    total_pnl = float(cdf["pnl"].sum()) if total_trades else 0.0
    end_capital = start_capital + total_pnl

    win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
    avg_trade_pnl = (total_pnl / total_trades) if total_trades else 0.0
    avg_win = float(cdf.loc[cdf["pnl"] > 0, "pnl"].mean()) if wins else 0.0
    avg_loss = float(cdf.loc[cdf["pnl"] <= 0, "pnl"].mean()) if losses else 0.0
    win_loss_ratio = (avg_win / abs(avg_loss)) if (avg_loss < 0) else np.nan
    best_trade = float(cdf["pnl"].max()) if total_trades else 0.0
    worst_trade = float(cdf["pnl"].min()) if total_trades else 0.0

    # durations and exposure
    if total_trades:
        dur = (pd.to_datetime(cdf["exit_date"]) - pd.to_datetime(cdf["entry_date"])).dt.days
        avg_dur = float(dur.mean())
        med_dur = float(dur.median())
        if len(daily) > 0:
            span_days = (daily["date"].iloc[-1] - daily["date"].iloc[0]).days + 1
        else:
            span_days = max(1, int(dur.sum()))
        time_in_mkt_pct = (dur.sum() / span_days) * 100.0 if span_days > 0 else 0.0
    else:
        avg_dur = med_dur = time_in_mkt_pct = 0.0

    # risk/return
    if len(daily) >= 2:
        r = daily["ret"].values
        mu = float(np.nanmean(r))
        sigma = float(np.nanstd(r, ddof=1))
        downside = float(np.nanstd(np.clip(r, a_max=0, a_min=None), ddof=1))
        ann = np.sqrt(252.0)
        sharpe = (mu / sigma) * ann if sigma > 0 else np.nan
        sortino = (mu / downside) * ann if downside > 0 else np.nan
        max_dd_abs = float(daily["drawdown"].max())
        max_dd_pct = (max_dd_abs / max(1e-9, daily["equity"].max())) * 100.0
        years = max(1e-9, (daily["date"].iloc[-1] - daily["date"].iloc[0]).days / 365.25)
        cagr = ( (end_capital / max(1e-9, start_capital)) ** (1/years) - 1 ) * 100.0 if start_capital > 0 else np.nan
        calmar = (cagr / (max_dd_pct if max_dd_pct>0 else np.nan)) if not np.isnan(cagr) else np.nan
        vol_annual = sigma * ann
    else:
        sharpe = sortino = calmar = vol_annual = np.nan
        max_dd_abs = max_dd_pct = 0.0
        cagr = np.nan

    summary = {
        "start_capital": round(start_capital,2),
        "end_capital": round(end_capital,2),
        "net_pnl": round(total_pnl,2),
        "total_trades": int(total_trades),
        "winners": int(wins),
        "losers": int(losses),
        "percent_profitable": round(win_rate,2),
        "avg_trade_pnl": round(avg_trade_pnl,2),
        "avg_win": round(avg_win,2),
        "avg_loss": round(avg_loss,2),
        "win_loss_ratio": round(win_loss_ratio,3) if not np.isnan(win_loss_ratio) else None,
        "gross_profit": round(gross_profit,2),
        "gross_loss": round(gross_loss,2),
        "profit_factor": round(profit_factor,3) if not np.isnan(profit_factor) else None,
        "best_trade": round(best_trade,2),
        "worst_trade": round(worst_trade,2),
        "avg_trade_duration_days": round(avg_dur,2),
        "median_trade_duration_days": round(med_dur,2),
        "time_in_market_pct": round(time_in_mkt_pct,2),
        "max_drawdown_abs": round(max_dd_abs,2),
        "max_drawdown_pct": round(max_dd_pct,2),
        "volatility_annualized": round(vol_annual,4) if not np.isnan(vol_annual) else None,
        "sharpe": round(sharpe,3) if not np.isnan(sharpe) else None,
        "sortino": round(sortino,3) if not np.isnan(sortino) else None,
        "cagr_pct": round(cagr,2) if not np.isnan(cagr) else None,
        "calmar": round(calmar,3) if not np.isnan(calmar) else None
    }
    return summary, cdf

# ---------------- Single run and sweep driver ----------------

def run_once(files, capital, risk_per_trade, atr_mult, adx_threshold, trail, tag):
    all_trades = []
    for f in files:
        symbol = os.path.basename(f).replace(".csv","")
        try:
            df = load_symbol_df(f)
        except Exception:
            continue
        trades = backtest_symbol(df, symbol, capital, risk_per_trade, atr_mult, adx_threshold, trail)
        all_trades.extend(trades)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_csv = os.path.join(REPORTS_DIR, f"ema_daily_trades_{tag}_{ts}.csv")
    daily_csv  = os.path.join(REPORTS_DIR, f"ema_daily_equity_{tag}_{ts}.csv")
    summary_csv= os.path.join(REPORTS_DIR, f"ema_daily_summary_{tag}_{ts}.csv")

    # Build analytics and save
    pd.DataFrame(all_trades).to_csv(trades_csv, index=False)
    daily = build_daily_equity(all_trades, start_capital=capital)
    daily.to_csv(daily_csv, index=False)
    summary, cdf = summarize(all_trades, daily, capital)
    srow = pd.DataFrame([{
        "atr_mult": atr_mult,
        "adx_threshold": adx_threshold,
        "trail": trail,
        **summary
    }])
    srow.to_csv(summary_csv, index=False)
    return summary, trades_csv, daily_csv, summary_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=200000)
    ap.add_argument("--risk_per_trade", type=float, default=0.01)
    ap.add_argument("--atr_mult", type=float, default=None, help="single ATR multiple (use with --adx)")
    ap.add_argument("--adx", type=float, default=None, help="single ADX threshold")
    ap.add_argument("--atr_grid", type=str, default=None, help="comma list, e.g., 1.5,2.0,2.5,3.0")
    ap.add_argument("--adx_grid", type=str, default=None, help="comma list, e.g., 20,25,30")
    ap.add_argument("--trail", type=str, default="true")
    ap.add_argument("--limit_symbols", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DAILY_DIR, "*.csv")))
    if args.limit_symbols > 0:
        files = files[:args.limit_symbols]
    trail = args.trail.lower() == "true"

    results = []

    # Sweep mode
    if args.atr_grid and args.adx_grid:
        atrs = [float(x) for x in args.atr_grid.split(",")]
        adxs = [float(x) for x in args.adx_grid.split(",")]
        for a in atrs:
            for d in adxs:
                tag = f"a{a}_d{d}_t{int(trail)}"
                summary, tcsv, dcsv, scsv = run_once(
                    files, args.capital, args.risk_per_trade, a, d, trail, tag
                )
                results.append({"atr_mult":a,"adx_threshold":d,"trail":trail, **summary, "trades_csv":tcsv,"daily_csv":dcsv,"summary_csv":scsv})
        combined = pd.DataFrame(results)
        combined_path = os.path.join(REPORTS_DIR, f"ema_daily_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        combined.to_csv(combined_path, index=False)
        print(f"Saved combined sweep: {combined_path}")
        print(combined.sort_values(["calmar","profit_factor","sharpe"], ascending=[False,False,False]).head(10).to_string(index=False))
        return

    # Single run mode
    if args.atr_mult is None or args.adx is None:
        raise SystemExit("Provide either --atr_mult & --adx (single run) or both --atr_grid & --adx_grid (sweep).")
    tag = f"a{args.atr_mult}_d{args.adx}_t{int(trail)}"
    summary, tcsv, dcsv, scsv = run_once(
        files, args.capital, args.risk_per_trade, float(args.atr_mult), float(args.adx), trail, tag
    )
    print("Summary:", summary)
    print("Trades CSV:", tcsv)
    print("Daily equity CSV:", dcsv)
    print("Summary CSV:", scsv)

if __name__ == "__main__":
    main()
