#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib

# -------- CONFIG --------
BASE_DIR  = "/root/falah-ai-bot"
DAILY_DIR = os.path.join(BASE_DIR, "swing_data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Your core symbol universe
SYMBOLS_TO_TEST = ["RELIANCE", "BHARTIARTL", "CIPLA", "BRITANNIA", "BPCL"]

USE_ML_CONFIRM = True   # Set False to use pure W%R
NIFTY_FILE = None  # optional for market regime

# Trade params
INITIAL_CAPITAL  = 1_000_000
POSITION_SIZE    = 100_000
PROFIT_TARGET    = 0.12
STOP_LOSS        = 0.05
TRAIL_TRIGGER    = 0.06
TRAIL_DISTANCE   = 0.025
TRANSACTION_COST = 0.001
MAX_POSITIONS    = 5
MAX_TRADES       = 200

# CNC cost model
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

# -------- INDICATORS --------
def add_indicators(df):
    df['ema200'] = ta.ema(df['close'], length=200)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    return df

def willr_signals(df):
    h14 = df['high'].rolling(14).max()
    l14 = df['low'].rolling(14).min()
    df['w%r'] = (h14 - df['close']) / (h14 - l14) * -100
    buy  = (df['w%r'] < -80) & (df['w%r'].shift(1) >= -80)
    sell = (df['w%r'] > -20) & (df['w%r'].shift(1) <= -20)
    df['signal'] = 0
    df.loc[buy, 'signal'] = 1
    df.loc[sell, 'signal'] = 0
    return df

def ml_signals(df, model):
    # Calculate ML features
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['volumechange'] = df['volume'].pct_change().fillna(0)
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    features = ['rsi','atr','adx','ema10','ema21','volumechange']
    df = df.dropna(subset=features).reset_index(drop=True)
    X = df[features]
    df['ml_signal'] = model.predict(X)
    return df

# -------- FILTER --------
def regime_filter(df, nifty_df=None):
    # Symbol regime: Uptrend, strong, high volume
    cond = (df['close'] > df['ema200']) & \
           (df['adx'] > 20) & \
           (df['volume'] > 1.5 * df['vol_sma20'])
    if nifty_df is not None:
        cond = cond & (nifty_df['close'] > nifty_df['ema200'])
    return df[cond].reset_index(drop=True)

# -------- BACKTEST --------
def backtest(df):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0

    for i in range(1, len(df)):
        date = df.at[i, 'date']
        price = df.at[i, 'close']
        sig = df.at[i, 'signal']

        # EXIT
        to_close = []
        for pid, pos in positions.items():
            days = (date - pos['entry_date']).days
            ret = (price - pos['entry_price']) / pos['entry_price']
            # trailing stop logic
            if price > pos['high']:
                pos['high'] = price
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = price * (1 - TRAIL_DISTANCE)
            if pos['trail_active']:
                new_stop = price * (1 - TRAIL_DISTANCE)
                if new_stop > pos['trail_stop']:
                    pos['trail_stop'] = new_stop
            # exit conditions
            if days >= 1 and (
                ret >= PROFIT_TARGET or ret <= -STOP_LOSS or sig == 0 or 
                (pos['trail_active'] and price <= pos['trail_stop'])
            ):
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                charges = calc_charges(buy_val, sell_val)
                exit_val = sell_val * (1 - TRANSACTION_COST)
                pnl = exit_val - buy_val - charges
                cash += exit_val
                trades.append({
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

    total_ret = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    df_tr = pd.DataFrame(trades)
    win_rate = (df_tr['pnl'] > 0).mean() * 100 if not df_tr.empty else 0
    avg_hold = df_tr['days_held'].mean() if not df_tr.empty else 0
    sharpe   = df_tr['return_pct'].mean() / df_tr['return_pct'].std() if not df_tr.empty and df_tr['return_pct'].std() != 0 else 0
    return total_ret, win_rate, avg_hold, sharpe, len(df_tr)

# -------- MAIN --------
if __name__ == "__main__":
    model = joblib.load(MODEL_PATH) if USE_ML_CONFIRM else None

    # Optional Nifty filter
    nifty_df = None
    if NIFTY_FILE is not None and os.path.exists(NIFTY_FILE):
    nifty_df = pd.read_csv(NIFTY_FILE, parse_dates=['date'])
    nifty_df = add_indicators(nifty_df)
    else:
        nifty_df = None
    results = {'return': [], 'win_rate': [], 'avg_hold': [],
               'sharpe': [], 'trades': []}

    for sym in SYMBOLS_TO_TEST:
        file_path = os.path.join(DAILY_DIR, f"{sym}.csv")
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = add_indicators(df)
        if USE_ML_CONFIRM:
            df = ml_signals(df, model)
        df = willr_signals(df)
        # combine with ML if set
        if USE_ML_CONFIRM and 'ml_signal' in df.columns:
            df['signal'] = np.where((df['signal'] == 1) & (df['ml_signal'] == 1), 1, 0)
        df = regime_filter(df, nifty_df)
        if df.empty:
            continue
        res = backtest(df)
        for k,v in zip(results.keys(), res):
            results[k].append(v)

    print("\nRegime-Filtered Daily Backtest Summary:")
    print(f"Avg Return:   {np.mean(results['return']):.2f}%")
    print(f"Avg Win Rate: {np.mean(results['win_rate']):.2f}%")
    print(f"Avg Hold:     {np.mean(results['avg_hold']):.2f} days")
    print(f"Sharpe:       {np.mean(results['sharpe']):.2f}")
    print(f"Total Trades: {np.sum(results['trades'])}")
