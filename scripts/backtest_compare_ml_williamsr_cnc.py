#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# ================
# CONFIGURATION
# ================
BASE_DIR       = "/root/falah-ai-bot"
DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1h':    os.path.join(BASE_DIR, "intraday_swing_data"),
    '15m':   os.path.join(BASE_DIR, "scalping_data"),
}

MODEL_PATH    = os.path.join(BASE_DIR, "model.pkl")
INITIAL_CAPITAL  = 1_000_000
POSITION_SIZE    = 100_000

PROFIT_TARGET    = 0.12     # 12%
STOP_LOSS        = 0.05     # 5%
TRAIL_TRIGGER    = 0.06     # 6%
TRAIL_DISTANCE   = 0.025    # 2.5%
TRANSACTION_COST = 0.001    # 0.1%
STT_RATE         = 0.001    # 0.1% buy + sell
STAMP_DUTY_RATE  = 0.00015  # 0.015% buy
EXCHANGE_RATE    = 0.0000345
GST_RATE         = 0.18
SEBI_RATE        = 0.000001
DP_CHARGE        = 13.5
MAX_POSITIONS    = 5
MAX_TRADES       = 200

# ================
# COST CALCULATION
# ================
def calc_charges(buy_val, sell_val):
    stt   = (buy_val + sell_val) * STT_RATE
    stamp = buy_val * STAMP_DUTY_RATE
    exch  = (buy_val + sell_val) * EXCHANGE_RATE
    gst   = exch * GST_RATE
    sebi  = (buy_val + sell_val) * SEBI_RATE
    dp    = DP_CHARGE
    return stt + stamp + exch + gst + sebi + dp

# ================
# ML SIGNALS
# ================
def ml_signals(df, model, features):
    df = df.copy()
    if not set(features).issubset(df.columns):
        print("   Warning: Not all ML features are present in data for symbol")
        return df
    df = df.dropna(subset=features)
    X = df[features]
    df.loc[X.index, 'signal'] = model.predict(X)
    # Reset index for backtest safety
    df = df.reset_index(drop=True)
    return df

# ================
# Williams %R SIGNALS
# ================
def willr_signals(df):
    df = df.copy()
    high14 = df['high'].rolling(14).max()
    low14 = df['low'].rolling(14).min()
    df['w%r'] = (high14 - df['close']) / (high14 - low14) * -100
    buy  = (df['w%r'] < -80) & (df['w%r'].shift(1) >= -80)
    sell = (df['w%r'] > -20) & (df['w%r'].shift(1) <= -20)
    df['signal'] = 0
    df.loc[buy, 'signal']  = 1
    df.loc[sell,'signal']  = 0
    df = df.dropna(subset=['w%r'])
    df = df.reset_index(drop=True)
    return df

# ================
# CNC BACKTEST ENGINE
# ================
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

            # trailing stop logic
            if price > pos['high']:
                pos['high'] = price
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop']   = price * (1 - TRAIL_DISTANCE)
            if pos['trail_active']:
                new_stop = price * (1 - TRAIL_DISTANCE)
                if new_stop > pos['trail_stop']:
                    pos['trail_stop'] = new_stop

            # exit conditions (≥1 day hold)
            if days >= 1 and (ret >= PROFIT_TARGET or ret <= -STOP_LOSS or sig == 0 or (pos['trail_active'] and price <= pos['trail_stop'])):
                buy_val   = pos['shares'] * pos['entry_price']
                sell_val  = pos['shares'] * price
                charges   = calc_charges(buy_val, sell_val)
                exit_val  = sell_val * (1 - TRANSACTION_COST)
                pnl       = exit_val - buy_val - charges
                cash     += exit_val
                trades.append({
                    'entry_date': pos['entry_date'], 'exit_date': date,
                    'entry_price': pos['entry_price'], 'exit_price': price,
                    'days_held': days, 'pnl': pnl, 'return_pct': ret*100,
                    'exit_reason': 'CNC Exit'
                })
                to_close.append(pid)
                trade_count += 1
                if trade_count >= MAX_TRADES:
                    break
        for pid in to_close:
            positions.pop(pid)
        if trade_count >= MAX_TRADES:
            break

        # ENTRY
        if sig == 1 and len(positions) < MAX_POSITIONS and cash >= POSITION_SIZE:
            shares = POSITION_SIZE / price
            cost   = POSITION_SIZE * (1 + TRANSACTION_COST)
            if cash >= cost:
                positions[len(positions)+1] = {
                    'entry_date': date, 'entry_price': price,
                    'shares': shares, 'high': price,
                    'trail_active': False, 'trail_stop': 0
                }
                cash -= cost

    # CLOSE REMAINING
    last_date  = df.iloc[-1]['date']
    last_price = df.iloc[-1]['close']
    for pos in positions.values():
        days = (last_date - pos['entry_date']).days
        if days >= 1:
            buy_val  = pos['shares'] * pos['entry_price']
            sell_val = pos['shares'] * last_price
            charges  = calc_charges(buy_val, sell_val)
            exit_val = sell_val * (1 - TRANSACTION_COST)
            pnl      = exit_val - buy_val - charges
            cash    += exit_val
            trades.append({
                'entry_date': pos['entry_date'], 'exit_date': last_date,
                'entry_price': pos['entry_price'], 'exit_price': last_price,
                'days_held': days,
                'pnl': pnl,
                'return_pct': ((last_price - pos['entry_price']) / pos['entry_price']) * 100,
                'exit_reason': 'EOD Exit'
            })

    df_tr = pd.DataFrame(trades)
    total_ret = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate  = (df_tr['pnl'] > 0).mean() * 100 if not df_tr.empty else 0
    avg_hold  = df_tr['days_held'].mean() if not df_tr.empty else 0
    sharpe    = df_tr['return_pct'].mean() / df_tr['return_pct'].std() if not df_tr.empty and df_tr['return_pct'].std() != 0 else 0

    return total_ret, win_rate, avg_hold, sharpe, len(df_tr)

# ================
# MAIN EXECUTION
# ================
if __name__ == "__main__":
    print("Loading ML model...")
    model = joblib.load(MODEL_PATH)
    # ML features your model expects - adjust if needed
    ml_features = ['rsi', 'atr', 'adx', 'ema_fast', 'ema_slow', 'volume_ratio']

    summary = []

    for tf, folder in DATA_DIRS.items():
        ml_agg = {'return': [], 'win_rate': [], 'avg_hold': [], 'sharpe': [], 'trades': []}
        wr_agg = {'return': [], 'win_rate': [], 'avg_hold': [], 'sharpe': [], 'trades': []}

        for fn in os.listdir(folder):
            if not fn.endswith('.csv'):
                continue
            path = os.path.join(folder, fn)
            df = pd.read_csv(path, parse_dates=['date'])
            if df.empty:
                continue

            # ML signals and backtest
            if set(ml_features).issubset(df.columns):
                df_ml = ml_signals(df, model, ml_features)
                res_ml = backtest(df_ml)
                for k, v in zip(ml_agg.keys(), res_ml):
                    ml_agg[k].append(v)
            else:
                print(f"ML features missing in {fn}, skipped ML backtest.")

            # Williams %R signals and backtest
            df_wr = willr_signals(df)
            if df_wr.empty:
                continue
            res_wr = backtest(df_wr)
            for k, v in zip(wr_agg.keys(), res_wr):
                wr_agg[k].append(v)

        # Summarize
        summary.append({
            'timeframe': tf,
            'ML Return (%)': np.mean(ml_agg['return']) if ml_agg['return'] else 0,
            'ML Win Rate (%)': np.mean(ml_agg['win_rate']) if ml_agg['win_rate'] else 0,
            'ML Avg Hold':     np.mean(ml_agg['avg_hold']) if ml_agg['avg_hold'] else 0,
            'ML Sharpe':       np.mean(ml_agg['sharpe']) if ml_agg['sharpe'] else 0,
            'ML Trades':       np.sum(ml_agg['trades']) if ml_agg['trades'] else 0,

            'WilliamsR Return (%)': np.mean(wr_agg['return']) if wr_agg['return'] else 0,
            'WilliamsR Win Rate (%)': np.mean(wr_agg['win_rate']) if wr_agg['win_rate'] else 0,
            'WilliamsR Avg Hold':     np.mean(wr_agg['avg_hold']) if wr_agg['avg_hold'] else 0,
            'WilliamsR Sharpe':       np.mean(wr_agg['sharpe']) if wr_agg['sharpe'] else 0,
            'WilliamsR Trades':       np.sum(wr_agg['trades']) if wr_agg['trades'] else 0,
        })

    df_summary = pd.DataFrame(summary).set_index('timeframe').round(2)
    print("\nBacktest Summary:")
    print(df_summary)
