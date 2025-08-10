# williams_r_cnc_backtest.py

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import timedelta

# -----------------------
# CONFIGURATION
# -----------------------
BASE_DIR         = "/root/falah-ai-bot"
DATA_DIR         = os.path.join(BASE_DIR, "swing_data")
INITIAL_CAPITAL  = 1_000_000      # ₹10 lakhs
POSITION_SIZE    = 100_000        # ₹1 lakh per trade
PROFIT_TARGET    = 0.12           # 12%
STOP_LOSS        = 0.05           # 5%
TRAIL_TRIGGER    = 0.06           # 6%
TRAIL_DISTANCE   = 0.025          # 2.5%
TRANSACTION_COST = 0.001          # 0.1% per side
STT_RATE         = 0.001          # 0.1% buy+sell
STAMP_DUTY_RATE  = 0.00015        # 0.015% buy
EXCHANGE_RATE    = 0.0000345      # per trade
GST_RATE         = 0.18           # on exchange fee
SEBI_RATE        = 0.000001       # per trade
DP_CHARGE        = 13.5           # ₹13.5 per sell
MAX_POSITIONS    = 5

# -----------------------
# CHARGES CALCULATION
# -----------------------
def calc_charges(buy_val, sell_val):
    stt     = (buy_val + sell_val) * STT_RATE
    stamp   = buy_val * STAMP_DUTY_RATE
    exch    = (buy_val + sell_val) * EXCHANGE_RATE
    gst     = exch * GST_RATE
    sebi    = (buy_val + sell_val) * SEBI_RATE
    dp      = DP_CHARGE
    return stt + stamp + exch + gst + sebi + dp

# -----------------------
# BACKTEST FUNCTION
# -----------------------
def backtest_symbol(filepath):
    df = pd.read_csv(filepath, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Calculate Williams %R
    df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    # Prepare variables
    cash       = INITIAL_CAPITAL
    positions  = {}
    trades     = []

    for i in range(14, len(df)):
        date  = df.at[i, "date"]
        price = df.at[i, "close"]
        wr    = df.at[i, "williams_r"]

        # Update and exit existing positions
        to_close = []
        for pid, pos in positions.items():
            days_held = (date - pos["entry_date"]).days
            ret       = (price - pos["entry_price"]) / pos["entry_price"]

            # Activate trailing stop
            if price > pos["high"]:
                pos["high"] = price
            if not pos["trail_active"] and ret >= TRAIL_TRIGGER:
                pos["trail_active"] = True
                pos["trail_stop"]   = price * (1 - TRAIL_DISTANCE)
            if pos["trail_active"]:
                new_stop = price * (1 - TRAIL_DISTANCE)
                if new_stop > pos["trail_stop"]:
                    pos["trail_stop"] = new_stop

            # Check exits
            exit_flag = False
            reason    = None
            if days_held >= 1:
                if ret >= PROFIT_TARGET:
                    exit_flag, reason = True, "Profit Target"
                elif ret <= -STOP_LOSS:
                    exit_flag, reason = True, "Stop Loss"
                elif wr > -20:
                    exit_flag, reason = True, "Overbought Exit"
                elif pos["trail_active"] and price <= pos["trail_stop"]:
                    exit_flag, reason = True, "Trailing Stop"

            if exit_flag:
                buy_val  = POSITION_SIZE
                sell_val = pos["shares"] * price
                charges  = calc_charges(buy_val, sell_val)
                exit_val = sell_val * (1 - TRANSACTION_COST)
                pnl      = exit_val - buy_val - charges
                cash    += exit_val
                trades.append({
                    "entry_date": pos["entry_date"], "exit_date": date,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "days_held": days_held, "pnl": pnl, "return_pct": ret*100,
                    "exit_reason": reason, "charges": charges
                })
                to_close.append(pid)

        for pid in to_close:
            del positions[pid]

        # Entry condition
        if wr < -80 and len(positions) < MAX_POSITIONS and cash >= POSITION_SIZE:
            shares = POSITION_SIZE / price
            positions[len(positions)+1] = {
                "entry_date": date, "entry_price": price, "shares": shares,
                "high": price, "trail_active": False, "trail_stop": 0
            }
            cash -= POSITION_SIZE * (1 + TRANSACTION_COST)

    # Close any remaining at last bar
    last = df.iloc[-1]
    for pos in positions.values():
        date  = last["date"]
        price = last["close"]
        days_held = (date - pos["entry_date"]).days
        if days_held >= 1:
            ret = (price - pos["entry_price"]) / pos["entry_price"]
            buy_val  = POSITION_SIZE
            sell_val = pos["shares"] * price
            charges  = calc_charges(buy_val, sell_val)
            exit_val = sell_val * (1 - TRANSACTION_COST)
            pnl      = exit_val - buy_val - charges
            cash    += exit_val
            trades.append({
                "entry_date": pos["entry_date"], "exit_date": date,
                "entry_price": pos["entry_price"], "exit_price": price,
                "days_held": days_held, "pnl": pnl, "return_pct": ret*100,
                "exit_reason": "End", "charges": charges
            })

    # Compile performance
    tr = pd.DataFrame(trades)
    total_ret   = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate    = (tr["pnl"] > 0).mean() * 100
    avg_hold    = tr["days_held"].mean()
    sharpe      = tr["return_pct"].mean() / tr["return_pct"].std()
    return total_ret, win_rate, avg_hold, sharpe, len(tr)

# -----------------------
# RUN BACKTEST OVER ALL SYMBOLS
# -----------------------
if __name__ == "__main__":
    results = []
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue
        fullpath = os.path.join(DATA_DIR, fname)
        ret, win, hold, shp, trades = backtest_symbol(fullpath)
        results.append((fname.replace(".csv",""), ret, win, hold, shp, trades))

    df_res = pd.DataFrame(results, columns=["symbol","return","win_rate","avg_hold","sharpe","trades"])
    df_res_summary = df_res.agg({
        "return":"mean","win_rate":"mean","avg_hold":"mean","sharpe":"mean","trades":"sum"
    })
    print("\nWilliams %R CNC Aggregate Performance:")
    print(df_res_summary.round(2))
