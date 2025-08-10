#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

# -----------------------
# CONFIGURATION
# -----------------------
BASE_DIR  = "/root/falah-ai-bot"
DAILY_DIR = os.path.join(BASE_DIR, "swing_data")

# ✅ Replace with the exact profitable list from your +35.9% run
SYMBOLS_TO_TEST = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    # Add ALL symbols from the original good test here
]

INITIAL_CAPITAL  = 1_000_000
POSITION_SIZE    = 100_000
PROFIT_TARGET    = 0.12
STOP_LOSS        = 0.05
TRAIL_TRIGGER    = 0.06
TRAIL_DISTANCE   = 0.025
TRANSACTION_COST = 0.001
MAX_POSITIONS    = 5
MAX_TRADES       = 250   # Limit total trades

# CNC cost parameters
STT_RATE         = 0.001
STAMP_DUTY_RATE  = 0.00015
EXCHANGE_RATE    = 0.0000345
GST_RATE         = 0.18
SEBI_RATE        = 0.000001
DP_CHARGE        = 13.5

def calc_charges(buy_val, sell_val):
    stt   = (buy_val + sell_val) * STT_RATE
    stamp = buy_val * STAMP_DUTY_RATE
    exch  = (buy_val + sell_val) * EXCHANGE_RATE
    gst   = exch * GST_RATE
    sebi  = (buy_val + sell_val) * SEBI_RATE
    return stt + stamp + exch + gst + sebi + DP_CHARGE

# -----------------------
# STRATEGY SIGNALS
# -----------------------
def willr_signals(df):
    high14 = df['high'].rolling(14).max()
    low14  = df['low'].rolling(14).min()
    df['w%r'] = (high14 - df['close']) / (high14 - low14) * -100
    buy  = (df['w%r'] < -80) & (df['w%r'].shift(1) >= -80)
    sell = (df['w%r'] > -20) & (df['w%r'].shift(1) <= -20)
    df['signal'] = 0
    df.loc[buy, 'signal'] = 1
    df.loc[sell, 'signal'] = 0
    df = df.dropna(subset=['w%r']).reset_index(drop=True)
    return df

# Trend filter: trade only if price > 200 EMA
def market_filter(df):
    df['ema200'] = df['close'].rolling(200).mean()
    return df[df['close'] > df['ema200']].copy().reset_index(drop=True)

# -----------------------
# BACKTEST ENGINE
# -----------------------
def backtest(df):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0

    for i in range(1, len(df)):
        date  = df.at[i, 'date']
        price = df.at[i, 'close']
        sig   = df.at[i, 'signal']

        # EXIT
        to_close = []
        for pid, pos in positions.items():
            days = (date - pos['entry_date']).days
            ret  = (price - pos['entry_price']) / pos['entry_price']
            if price > pos['high']:
                pos['high'] = price
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop']   = price * (1 - TRAIL_DISTANCE)
            if pos['trail_active']:
                new_stop = price * (1 - TRAIL_DISTANCE)
                if new_stop > pos['trail_stop']:
                    pos['trail_stop'] = new_stop
            if days >= 1 and (
                ret >= PROFIT_TARGET or ret <= -STOP_LOSS or sig == 0 or
                (pos['trail_active'] and price <= pos['trail_stop'])
            ):
                buy_val   = pos['shares'] * pos['entry_price']
                sell_val  = pos['shares'] * price
                charges   = calc_charges(buy_val, sell_val)
                exit_val  = sell_val * (1 - TRANSACTION_COST)
                pnl       = exit_val - buy_val - charges
                cash     += exit_val
                trades.append({
                    'entry_date': pos['entry_date'], 'exit_date': date,
                    'pnl': pnl, 'return_pct': ret*100, 'days_held': days
                })
                to_close.append(pid)
                trade_count += 1
        for pid in to_close:
            positions.pop(pid)

        if trade_count >= MAX_TRADES:
            break

        # ENTRY
        if sig == 1 and len(positions) < MAX_POSITIONS and cash >= POSITION_SIZE:
            shares = POSITION_SIZE / price
            cost_val = POSITION_SIZE * (1 + TRANSACTION_COST)
            if cash >= cost_val:
                positions[len(positions)+1] = {
                    'entry_date': date, 'entry_price': price,
                    'shares': shares, 'high': price,
                    'trail_active': False, 'trail_stop': 0
                }
                cash -= cost_val

    # Close remaining
    last_date = df.iloc[-1]['date']
    last_price = df.iloc[-1]['close']
    for pos in positions.values():
        days = (last_date - pos['entry_date']).days
        if days >= 1:
            buy_val  = pos['shares'] * pos['entry_price']
            sell_val = pos['shares'] * last_price
            charges  = calc_charges(buy_val, sell_val)
            exit_val = sell_val * (1 - TRANSACTION_COST)
            pnl      = exit_val - buy_val - charges
            cash += exit_val
            trades.append({
                'entry_date': pos['entry_date'], 'exit_date': last_date,
                'pnl': pnl,
                'return_pct': ((last_price - pos['entry_price']) / pos['entry_price']) * 100,
                'days_held': days
            })

    df_tr = pd.DataFrame(trades)
    total_ret = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate  = (df_tr['pnl'] > 0).mean() * 100 if not df_tr.empty else 0
    avg_hold  = df_tr['days_held'].mean() if not df_tr.empty else 0
    sharpe    = df_tr['return_pct'].mean() / df_tr['return_pct'].std() if not df_tr.empty and df_tr['return_pct'].std() != 0 else 0

    return total_ret, win_rate, avg_hold, sharpe, len(df_tr)

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    results = {'return': [], 'win_rate': [], 'avg_hold': [], 'sharpe': [], 'trades': []}

    for fn in os.listdir(DAILY_DIR):
        sym = fn.replace('.csv','')
        if sym not in SYMBOLS_TO_TEST:
            continue
        df = pd.read_csv(os.path.join(DAILY_DIR, fn), parse_dates=['date'])
        if df.empty: 
            continue
        df = willr_signals(df)
        df = market_filter(df)
        if df.empty:
            continue
        r = backtest(df)
        for k,v in zip(results.keys(), r):
            results[k].append(v)

    print("\nFiltered Daily W%R CNC Results:")
    print(f"Avg Return:   {np.mean(results['return']):.2f}%")
    print(f"Avg Win Rate: {np.mean(results['win_rate']):.2f}%")
    print(f"Avg Hold:     {np.mean(results['avg_hold']):.2f} days")
    print(f"Sharpe:       {np.mean(results['sharpe']):.2f}")
    print(f"Total Trades: {np.sum(results['trades'])}")
