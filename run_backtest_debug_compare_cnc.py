# run_backtest_debug_compare_cnc.py

import pandas as pd
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

# Clean & filter
df = df[(df["close"] >= 50) & (df["close"] <= 10_000)].copy()
df.sort_values("date", inplace=True)
df.reset_index(drop=True, inplace=True)

# Generate ML signals
features = [c for c in ["rsi","atr","adx","ema10","ema21","volumechange"] if c in df]
df.dropna(subset=features, inplace=True)
df["signal"] = model.predict(df[features])

# CNC backtest
results, cash, positions = [], INITIAL_CAPITAL, {}
for i in range(len(df)):
    row = df.iloc[i]
    date, price, signal = row["date"], row["close"], row["signal"]
    # Exit logic
    to_close = []
    for pid, pos in positions.items():
        days = (date - pos["entry_date"]).days
        ret  = (price - pos["entry_price"]) / pos["entry_price"]
        # Trailing stop
        if price > pos["high"] : pos["high"] = price
        if not pos["trail_active"] and ret >= pos["profit_target"]*0.5:
            pos["trail_active"] = True
            pos["trail_stop"]   = price*(1-pos["trail_dist"])
        if pos["trail_active"]:
            new_stop = price*(1-pos["trail_dist"])
            if new_stop>pos["trail_stop"]: pos["trail_stop"]=new_stop
        # CNC exit conditions
        if days>=MIN_HOLD_DAYS and (
           days>=MAX_HOLD_DAYS or
           (pos["trail_active"] and price<=pos["trail_stop"]) or
           ret<=-pos["stop_loss"] or
           ret>=pos["profit_target"] or
           signal==0):
            # calculate charges
            buy_val  = POSITION_SIZE
            sell_val = pos["shares"]*price
            stt      = (buy_val+sell_val)*STT_RATE
            exchange = (buy_val+sell_val)*0.0000345
            gst      = exchange*0.18
            sebi     = (buy_val+sell_val)*0.000001
            stamp    = buy_val*STAMP_DUTY_RATE
            dp       = DP_CHARGE
            total_charges = stt+exchange+gst+sebi+stamp+dp
            exit_val = pos["shares"]*price*(1-TRANSACTION_COST_RATE)
            pnl      = exit_val - buy_val - total_charges
            cash    += exit_val
            results.append({
                "entry_date": pos["entry_date"],
                "exit_date" : date,
                "entry_price": pos["entry_price"],
                "exit_price" : price,
                "days_held"  : days,
                "pnl"        : pnl,
                "return_pct" : ret*100,
                "charges"    : total_charges,
                "exit_reason": pos.get("exit_reason","CNC Exit")
            })
            to_close.append(pid)
    for pid in to_close: positions.pop(pid)
    # Entry logic
    if signal==1 and len(positions)<5 and cash>=POSITION_SIZE*1.5:
        shares = POSITION_SIZE/price
        positions[len(positions)+1] = {
            "entry_date": date,
            "entry_price": price,
            "shares": shares,
            "high": price,
            "profit_target": 0.15,
            "stop_loss": 0.05,
            "trail_active": False,
            "trail_stop": 0,
            "trail_dist": 0.03
        }
        cash -= POSITION_SIZE

# Finalize any open positions at last date
last_date, last_price = df.iloc[-1]["date"], df.iloc[-1]["close"]
for pos in positions.values():
    days = (last_date-pos["entry_date"]).days
    if days>=MIN_HOLD_DAYS:
        ret = (last_price-pos["entry_price"])/pos["entry_price"]
        buy_val  = POSITION_SIZE
        sell_val = pos["shares"]*last_price
        stt      = (buy_val+sell_val)*STT_RATE
        exchange = (buy_val+sell_val)*0.0000345
        gst      = exchange*0.18
        sebi     = (buy_val+sell_val)*0.000001
        stamp    = buy_val*STAMP_DUTY_RATE
        dp       = DP_CHARGE
        total_charges = stt+exchange+gst+sebi+stamp+dp
        exit_val = pos["shares"]*last_price*(1-TRANSACTION_COST_RATE)
        pnl      = exit_val - buy_val - total_charges
        cash    += exit_val
        results.append({
            "entry_date": pos["entry_date"],
            "exit_date" : last_date,
            "entry_price": pos["entry_price"],
            "exit_price" : last_price,
            "days_held"  : days,
            "pnl"        : pnl,
            "return_pct" : ret*100,
            "charges"    : total_charges,
            "exit_reason": "End of Period"
        })

# Analyze performance
trades_df = pd.DataFrame(results)
total_return = (cash-INITIAL_CAPITAL)/INITIAL_CAPITAL*100
win_rate     = (trades_df["pnl"]>0).mean()*100
avg_hold     = trades_df["days_held"].mean()
sharpe       = trades_df["return_pct"].mean()/trades_df["return_pct"].std()

print(f"CNC ML Strategy Results:")
print(f"Total Trades: {len(trades_df)}, Win Rate: {win_rate:.2f}%")
print(f"Total Return: {total_return:.2f}%, Avg Hold: {avg_hold:.1f} days, Sharpe: {sharpe:.2f}")
