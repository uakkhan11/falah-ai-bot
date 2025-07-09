# backtest_engine.py

import os
import pandas as pd
import numpy as np
from datetime import datetime

# SETTINGS
SYMBOLS = ["INFY", "TCS", "HDFCBANK"]
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"
INITIAL_CAPITAL = 1_000_000  # 10 lakh
RISK_PER_TRADE = 0.02  # 2%
TARGET_MULTIPLIER = 3
DATA_DIR = "/root/falah-ai-bot/historical_data"

results = []
equity_curve = []

capital = INITIAL_CAPITAL
peak = INITIAL_CAPITAL
drawdowns = []

for symbol in SYMBOLS:
    file = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(file):
        print(f"⚠️ Missing data for {symbol}")
        continue

    df = pd.read_csv(file, parse_dates=["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].reset_index(drop=True)

    trades = []
    in_trade = False

    for i in range(20, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Example entry rule: Close above 20 SMA
        sma = df["close"].iloc[i - 20:i].mean()
        if not in_trade and row["close"] > sma:
            entry_price = row["close"]
            atr = df["close"].iloc[i - 14:i].std()
            sl = entry_price - atr * 1.5
            target = entry_price + (entry_price - sl) * TARGET_MULTIPLIER
            qty = int((capital * RISK_PER_TRADE) / (entry_price - sl))
            if qty <= 0:
                continue
            in_trade = True
            entry_date = row["date"]
            continue

        if in_trade:
            exit_price = None
            reason = ""
            # Stop loss hit
            if row["low"] <= sl:
                exit_price = sl
                reason = "Stop Loss"
            # Target hit
            elif row["high"] >= target:
                exit_price = target
                reason = "Target"
            # Exit at end of data
            elif i == len(df) - 2:
                exit_price = row["close"]
                reason = "EOD Exit"

            if exit_price:
                pnl = (exit_price - entry_price) * qty
                capital += pnl
                peak = max(peak, capital)
                drawdown = (peak - capital) / peak
                drawdowns.append(drawdown)

                trades.append({
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": row["date"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "reason": reason
                })

                equity_curve.append({
                    "date": row["date"],
                    "capital": capital
                })

                in_trade = False

    if trades:
        tdf = pd.DataFrame(trades)
        win_rate = (tdf["pnl"] > 0).mean() * 100
        avg_pnl = tdf["pnl"].mean()
        results.append({
            "symbol": symbol,
            "trades": len(tdf),
            "win_rate": win_rate,
            "avg_pnl": avg_pnl
        })
        tdf.to_csv(f"backtest_{symbol}.csv", index=False)
        print(f"✅ {symbol}: {len(tdf)} trades | Win Rate {win_rate:.1f}% | Avg PnL ₹{avg_pnl:.2f}")
    else:
        print(f"⚠️ No trades triggered for {symbol}")

# Summary
summary = pd.DataFrame(results)
summary.to_csv("backtest_summary.csv", index=False)

# Equity curve
ec = pd.DataFrame(equity_curve)
if not ec.empty:
    ec.to_csv("equity_curve.csv", index=False)
    cagr = ((capital / INITIAL_CAPITAL) ** (1 / (len(ec) / 252))) - 1
    max_dd = max(drawdowns) * 100
    ret_series = ec["capital"].pct_change().dropna()
    sharpe = ret_series.mean() / ret_series.std() * np.sqrt(252)
    print(f"🎯 Backtest Complete | CAGR: {cagr:.2%} | Max DD: {max_dd:.2f}% | Sharpe: {sharpe:.2f}")
else:
    print("⚠️ No equity data to compute performance metrics.")
