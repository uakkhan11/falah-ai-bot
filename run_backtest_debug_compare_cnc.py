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
TAKE_PROFIT_PCT        = 0.15           # 15%
INITIAL_STOP_LOSS_PCT  = 0.05           # 5%
# ======================

# 1. Load ML signals
df_ml = pd.read_csv(CSV_PATH, parse_dates=["date"])
df_ml.columns = df_ml.columns.str.lower()
model = joblib.load(MODEL_PATH)

# 2. Load raw OHLC data and merge signals
df_raw = pd.read_csv("swing_data/RELIANCE.csv", parse_dates=["date"])
df_raw.columns = df_raw.columns.str.lower()
df_raw["date"] = df_raw["date"].dt.tz_localize(None)

# Right after loading:
df_ml["date"]  = pd.to_datetime(df_ml["date"], errors="coerce").dt.tz_localize(None)
df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce").dt.tz_localize(None)

df = df_raw.merge(df_ml[["date"]], on="date", how="inner")
df.reset_index(drop=True, inplace=True)

# 3. Calculate indicators for ML features
df["rsi"]         = ta.rsi(df["close"], length=14)
df["atr"]         = ta.atr(df["high"], df["low"], df["close"], length=14)
df["adx"]         = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
df["ema10"]       = ta.ema(df["close"], length=10)
df["ema21"]       = ta.ema(df["close"], length=21)
macd             = ta.macd(df["close"], fast=12, slow=26, signal=9)
df["macd_hist"]   = macd["MACDh_12_26_9"]
df["macd_signal"] = macd["MACDs_12_26_9"]

# 4. Generate ML signals
features = ["rsi","atr","adx","ema10","ema21","macd_hist","macd_signal"]
df.dropna(subset=features, inplace=True)
X = df[features]
df["signal"] = model.predict(X)

# 5. CNC backtest
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
        ret  = (price - pos["entry_price"])/pos["entry_price"]
        if price>pos["high"]:
            pos["high"] = price
        if not pos["trail_active"] and ret>=TAKE_PROFIT_PCT*0.5:
            pos["trail_active"] = True
            pos["trail_stop"]   = price*(1-pos["trail_dist"])
        if pos["trail_active"]:
            new_stop = price*(1-pos["trail_dist"])
            if new_stop>pos["trail_stop"]:
                pos["trail_stop"] = new_stop

        if days>=MIN_HOLD_DAYS and (
           days>=MAX_HOLD_DAYS or
           (pos["trail_active"] and price<=pos["trail_stop"]) or
           ret<=-INITIAL_STOP_LOSS_PCT or
           ret>=TAKE_PROFIT_PCT or
           signal==0):
            buy_val  = POSITION_SIZE
            sell_val = pos["shares"]*price
            stt      = (buy_val+sell_val)*STT_RATE
            exc      = (buy_val+sell_val)*0.0000345
            gst      = exc*0.18
            sebi     = (buy_val+sell_val)*0.000001
            stamp    = buy_val*STAMP_DUTY_RATE
            dp       = DP_CHARGE
            charges  = stt+exc+gst+sebi+stamp+dp
            exit_val = pos["shares"]*price*(1-TRANSACTION_COST_RATE)
            pnl      = exit_val - buy_val - charges
            cash    += exit_val
            results.append({
                "entry_date": pos["entry_date"],
                "exit_date" : date,
                "entry_price": pos["entry_price"],
                "exit_price" : price,
                "days_held"  : days,
                "pnl"        : pnl,
                "return_pct" : ret*100,
                "charges"    : charges,
                "exit_reason": "CNC Exit"
            })
            to_close.append(pid)
    for pid in to_close:
        positions.pop(pid)

    # Entry logic
    if signal==1 and len(positions)<5 and cash>=POSITION_SIZE*1.5:
        shares = POSITION_SIZE/price
        positions[len(positions)+1] = {
            "entry_date"   : date,
            "entry_price"  : price,
            "shares"       : shares,
            "high"         : price,
            "trail_active" : False,
            "trail_stop"   : 0,
            "trail_dist"   : 0.03
        }
        cash -= POSITION_SIZE

# Final close
last = df.iloc[-1]
for pos in positions.values():
    days = (last["date"]-pos["entry_date"]).days
    if days>=MIN_HOLD_DAYS:
        ret      = (last["close"]-pos["entry_price"])/pos["entry_price"]
        buy_val  = POSITION_SIZE
        sell_val = pos["shares"]*last["close"]
        stt      = (buy_val+sell_val)*STT_RATE
        exc      = (buy_val+sell_val)*0.0000345
        gst      = exc*0.18
        sebi     = (buy_val+sell_val)*0.000001
        stamp    = buy_val*STAMP_DUTY_RATE
        dp       = DP_CHARGE
        charges  = stt+exc+gst+sebi+stamp+dp
        exit_val = pos["shares"]*last["close"]*(1-TRANSACTION_COST_RATE)
        pnl      = exit_val - buy_val - charges
        cash    += exit_val
        results.append({
            "entry_date": pos["entry_date"],
            "exit_date" : last["date"],
            "entry_price": pos["entry_price"],
            "exit_price" : last["close"],
            "days_held"  : days,
            "pnl"        : pnl,
            "return_pct" : ret*100,
            "charges"    : charges,
            "exit_reason": "End of Period"
        })

# Performance summary
trades_df = pd.DataFrame(results)
total_ret = (cash-INITIAL_CAPITAL)/INITIAL_CAPITAL*100
win_rate  = (trades_df["pnl"]>0).mean()*100
avg_hold  = trades_df["days_held"].mean()
sharpe    = trades_df["return_pct"].mean()/trades_df["return_pct"].std()

print(f"\nCNC ML Strategy Results:")
print(f"Total Trades: {len(trades_df)}, Win Rate: {win_rate:.2f}%")
print(f"Total Return: {total_ret:.2f}%, Avg Hold: {avg_hold:.1f} days, Sharpe: {sharpe:.2f}")
