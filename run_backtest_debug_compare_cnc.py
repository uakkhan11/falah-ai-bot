# run_backtest_debug_compare_cnc.py

import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# ======================
# CNC ML Strategy Configuration
# ======================
CSV_PATH               = "your_training_data.csv"
MODEL_PATH             = "model.pkl"
INITIAL_CAPITAL        = 1_000_000      # ₹10 Lakhs
POSITION_SIZE          = 100_000        # ₹1 Lakh per trade
MIN_HOLD_DAYS          = 1              # CNC compliance
MAX_HOLD_DAYS          = 20
TRANSACTION_COST_RATE  = 0.001          # 0.1% per side
STT_RATE               = 0.001          # 0.1% each side
STAMP_DUTY_RATE        = 0.00015        # 0.015% buy
DP_CHARGE              = 13.5           # ₹13.5 per sell
# ======================

# Load data and model
df = pd.read_csv(CSV_PATH, parse_dates=["date"])
model = joblib.load(MODEL_PATH)

# Clean & filter price outliers
df = df[(df["close"] >= 50) & (df["close"] <= 10_000)].copy()
df.sort_values("date", inplace=True)
df.reset_index(drop=True, inplace=True)

# Calculate technical indicators for ML features
df["rsi"]        = ta.rsi(df["close"], length=14)
df["atr"]        = ta.atr(df["high"], df["low"], df["close"], length=14)
df["adx"]        = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
df["ema10"]      = ta.ema(df["close"], length=10)
df["ema21"]      = ta.ema(df["close"], length=21)
macd            = ta.macd(df["close"], fast=12, slow=26, signal=9)
df["macd_hist"]  = macd["MACDh_12_26_9"]
df["macd_signal"]= macd["MACDs_12_26_9"]

# Prepare ML features
features = ["rsi","atr","adx","ema10","ema21","macd_hist","macd_signal"]
available = [f for f in features if f in df.columns]
if not available:
    raise ValueError(f"No valid features found. Columns: {df.columns.tolist()}")

df.dropna(subset=available, inplace=True)
df.reset_index(drop=True, inplace=True)

# Generate ML signals
X = df[available]
df["signal"] = model.predict(X)
df["ml_probability"] = model.predict_proba(X)[:,1]

# CNC backtest
results, cash, positions = [], INITIAL_CAPITAL, {}
for i in range(1, len(df)):
    row    = df.iloc[i]
    date   = row["date"]
    price  = row["close"]
    signal = row["signal"]
    # Exit logic
    to_close = []
    for pid, pos in positions.items():
        days = (date - pos["entry_date"]).days
        ret  = (price - pos["entry_price"]) / pos["entry_price"]
        # Trailing stop update
        if price > pos["high"]:
            pos["high"] = price
        if not pos["trail_active"] and ret >= pos["profit_target"] * 0.5:
            pos["trail_active"] = True
            pos["trail_stop"]   = price * (1 - pos["trail_dist"])
        if pos["trail_active"]:
            new_stop = price * (1 - pos["trail_dist"])
            if new_stop > pos["trail_stop"]:
                pos["trail_stop"] = new_stop
        # CNC exit conditions
        if days >= MIN_HOLD_DAYS and (
           days >= MAX_HOLD_DAYS or
           (pos["trail_active"] and price <= pos["trail_stop"]) or
           ret <= -pos["stop_loss"] or
           ret >= pos["profit_target"] or
           signal == 0):
            # Calculate charges
            buy_val  = POSITION_SIZE
            sell_val = pos["shares"] * price
            stt      = (buy_val + sell_val) * STT_RATE
            exc      = (buy_val + sell_val) * 0.0000345
            gst      = exc * 0.18
            sebi     = (buy_val + sell_val) * 0.000001
            stamp    = buy_val * STAMP_DUTY_RATE
            dp       = DP_CHARGE
            charges  = stt + exc + gst + sebi + stamp + dp
            exit_val = pos["shares"] * price * (1 - TRANSACTION_COST_RATE)
            pnl      = exit_val - buy_val - charges
            cash    += exit_val
            results.append({
                "entry_date": pos["entry_date"],
                "exit_date" : date,
                "entry_price": pos["entry_price"],
                "exit_price" : price,
                "days_held"  : days,
                "pnl"        : pnl,
                "return_pct" : ret * 100,
                "charges"    : charges,
                "exit_reason": "CNC Exit"
            })
            to_close.append(pid)
    for pid in to_close:
        positions.pop(pid)
    # Entry logic
    if signal == 1 and len(positions) < 5 and cash >= POSITION_SIZE * 1.5:
        shares = POSITION_SIZE / price
        positions[len(positions)+1] = {
            "entry_date"    : date,
            "entry_price"   : price,
            "shares"        : shares,
            "high"          : price,
            "profit_target" : TAKE_PROFIT_PCT,
            "stop_loss"     : INITIAL_STOP_LOSS_PCT,
            "trail_active"  : False,
            "trail_stop"    : 0,
            "trail_dist"    : 0.03
        }
        cash -= POSITION_SIZE

# Final close of any open positions
last = df.iloc[-1]
for pos in positions.values():
    days = (last["date"] - pos["entry_date"]).days
    if days >= MIN_HOLD_DAYS:
        ret      = (last["close"] - pos["entry_price"]) / pos["entry_price"]
        buy_val  = POSITION_SIZE
        sell_val = pos["shares"] * last["close"]
        stt      = (buy_val + sell_val) * STT_RATE
        exc      = (buy_val + sell_val) * 0.0000345
        gst      = exc * 0.18
        sebi     = (buy_val + sell_val) * 0.000001
        stamp    = buy_val * STAMP_DUTY_RATE
        dp       = DP_CHARGE
        charges  = stt + exc + gst + sebi + stamp + dp
        exit_val = pos["shares"] * last["close"] * (1 - TRANSACTION_COST_RATE)
        pnl      = exit_val - buy_val - charges
        cash    += exit_val
        results.append({
            "entry_date": pos["entry_date"],
            "exit_date" : last["date"],
            "entry_price": pos["entry_price"],
            "exit_price" : last["close"],
            "days_held"  : days,
            "pnl"        : pnl,
            "return_pct" : ret * 100,
            "charges"    : charges,
            "exit_reason": "End of Period"
        })

# Performance summary
trades_df   = pd.DataFrame(results)
total_ret   = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
win_rate    = (trades_df["pnl"] > 0).mean() * 100
avg_hold    = trades_df["days_held"].mean()
sharpe      = trades_df["return_pct"].mean() / trades_df["return_pct"].std()

print(f"\nCNC ML Strategy Results:")
print(f"Total Trades: {len(trades_df)}, Win Rate: {win_rate:.2f}%")
print(f"Total Return: {total_ret:.2f}%, Avg Hold: {avg_hold:.1f} days, Sharpe: {sharpe:.2f}")
