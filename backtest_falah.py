# backtest_falah.py

import os
import glob
import json
import numpy as np   # ✅ Fix: import numpy
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect

# ─── CONFIG ─────────────────────────────────────────────────────────
DATA_DIR = "/root/falah-ai-bot/historical_data"
RESULTS_DIR = "./backtest_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Automatically detect all CSVs in the folder
SYMBOLS = [os.path.basename(f).replace(".csv","") for f in glob.glob(f"{DATA_DIR}/*.csv")]
START_DATE = "2019-01-01"
END_DATE = "2023-12-31"
INITIAL_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.02
SLIPPAGE = 0.001
COMMISSION = 0.0005

# ─── LOAD ALL DATA ───────────────────────────────────────────────────
print("📂 Loading historical data...")
all_data = {}
for sym in SYMBOLS:
    path = os.path.join(DATA_DIR, f"{sym}.csv")
    if not os.path.exists(path):
        print(f"⚠️ Missing data for {sym}")
        continue
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date")
    df = df.loc[START_DATE:END_DATE]
    all_data[sym] = df
print(f"✅ Loaded data for {len(all_data)} symbols.")

ema_ok = 0
rsi_ok = 0
ai_ok = 0
all_ok = 0


# ─── MAIN BACKTEST LOOP ──────────────────────────────────────────────
capital = INITIAL_CAPITAL
peak = capital
drawdowns = []
equity_curve = []
trades = []

# Import your smart_scanner
from smart_scanner import model as ml_model
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# ✅ Define feature names to match ML training
FEATURE_NAMES = ["rsi", "ema10", "ema21", "atr", "vol_ratio"]

print("🚀 Running backtest...")
for sym, df in all_data.items():
    in_trade = False
    entry_price = sl = tp = None
    qty = 0

    for i in range(21, len(df)):
        date = df.index[i]
        today = df.iloc[i]

        # Compute indicators
        rsi_series = RSIIndicator(close=df["close"].iloc[:i+1], window=14).rsi()
        rsi = rsi_series.iloc[-1]

        ema10_series = EMAIndicator(close=df["close"].iloc[:i+1], window=10).ema_indicator()
        ema10 = ema10_series.iloc[-1]

        ema21_series = EMAIndicator(close=df["close"].iloc[:i+1], window=21).ema_indicator()
        ema21 = ema21_series.iloc[-1]

        atr_series = AverageTrueRange(
            high=df["high"].iloc[:i+1],
            low=df["low"].iloc[:i+1],
            close=df["close"].iloc[:i+1],
            window=14
        ).average_true_range()
        atr = atr_series.iloc[-1]

        rolling_mean = df["volume"].iloc[:i+1].rolling(10).mean().iloc[-1]
        if pd.isna(rolling_mean) or rolling_mean == 0:
            continue
        vol_ratio = today["volume"] / rolling_mean

        # ML features
        features_df = pd.DataFrame([[rsi, ema10, ema21, atr, vol_ratio]], columns=FEATURE_NAMES)

        prob = ml_model.predict_proba(features_df)[0][1]
        ai_score = prob * 5.0

# Entry criteria
ema_pass = ema10 > ema21
rsi_pass = rsi > 50
ai_pass = ai_score >= 2.5

if ema_pass:
    ema_ok += 1
if rsi_pass:
    rsi_ok += 1
if ai_pass:
    ai_ok += 1
if ema_pass and rsi_pass and ai_pass:
    all_ok += 1

entry_signal = ema_pass and rsi_pass and ai_pass


        if not in_trade and entry_signal:
            entry_price = today["close"]
            sl = entry_price - 1.5 * atr
            tp = entry_price + (entry_price - sl) * 3
            risk_amount = capital * RISK_PER_TRADE
            qty = int(risk_amount / (entry_price - sl))
            if qty < 1:
                continue
            in_trade = True
            entry_date = date
            continue

        if in_trade:
            exit_price = None
            reason = ""
            if today["low"] <= sl:
                exit_price = sl
                reason = "Stop Loss"
            elif today["high"] >= tp:
                exit_price = tp
                reason = "Target Hit"
            elif i == len(df) - 1:
                exit_price = today["close"]
                reason = "End of Data"

            if exit_price:
                buy_price = entry_price * (1 + SLIPPAGE)
                sell_price = exit_price * (1 - SLIPPAGE)
                cost = buy_price * qty * (1 + COMMISSION)
                proceeds = sell_price * qty * (1 - COMMISSION)
                pnl = proceeds - cost

                capital += pnl
                peak = max(peak, capital)
                dd = (peak - capital) / peak
                drawdowns.append(dd)

                trades.append({
                    "symbol": sym,
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "reason": reason,
                    "qty": qty
                })

                equity_curve.append({"date": date, "capital": capital})

                in_trade = False

# ─── SAVE RESULTS ────────────────────────────────────────────────────
if trades:
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(os.path.join(RESULTS_DIR, "trades.csv"), index=False)
    print(f"✅ Saved {len(trades)} trades.")
else:
    print("⚠️ No trades recorded.")

if equity_curve:
    ec = pd.DataFrame(equity_curve)
    ec.to_csv(os.path.join(RESULTS_DIR, "equity_curve.csv"), index=False)
    returns = ec["capital"].pct_change().dropna()
    cagr = ((capital / INITIAL_CAPITAL) ** (1 / ((ec['date'].iloc[-1] - ec['date'].iloc[0]).days / 365.25))) - 1
    sharpe = returns.mean() / returns.std() * (252 ** 0.5)
    max_dd = max(drawdowns) * 100
    print(f"\n🎯 Backtest Complete")
    print(f"CAGR: {cagr:.2%}  |  Sharpe: {sharpe:.2f}  |  Max DD: {max_dd:.1f}%")
else:
    print("⚠️ No equity data to compute performance metrics.")

print(f"\nℹ️  Entry Filter Stats:")
print(f"EMA Passed: {ema_ok}")
print(f"RSI Passed: {rsi_ok}")
print(f"AI Score Passed: {ai_ok}")
print(f"All Conditions Passed: {all_ok}")
