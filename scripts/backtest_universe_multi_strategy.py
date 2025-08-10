#!/usr/bin/env python3
import os, pandas as pd, numpy as np
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta

# ===== CONFIGURATION =====
BASE_DIR = "/root/falah-ai-bot"
DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
}
TIMEFRAME = 'daily'   # change to '15minute' or '1hour' as needed
DATA_PATH = DATA_DIRS[TIMEFRAME]

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TRADE_LOG_CSV = os.path.join(BASE_DIR, f"backtest_trade_log_{TIMEFRAME}.csv")
USE_ML_CONFIRM = True   # Toggle AI confirmation

# Strategy Parameters
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE   = 100_000
PROFIT_TARGET   = 0.10    # Relaxed target from 12%
STOP_LOSS       = 0.05
TRAIL_TRIGGER   = 0.05
TRAIL_DISTANCE  = 0.02

# Relaxed Filters
ADX_THRESHOLD   = 15
VOLUME_MULTIPLIER = 1.2

TRANSACTION_COST= 0.001
MAX_POSITIONS   = 5
MAX_TRADES      = 2000  # Universe-wide

# CNC Charges
STT_RATE        = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_RATE   = 0.0000345
GST_RATE        = 0.18
SEBI_RATE       = 0.000001
DP_CHARGE       = 13.5

def calc_charges(buy_val, sell_val):
    stt = (buy_val + sell_val) * STT_RATE
    stamp = buy_val * STAMP_DUTY_RATE
    exch = (buy_val + sell_val) * EXCHANGE_RATE
    gst = exch * GST_RATE
    sebi = (buy_val + sell_val) * SEBI_RATE
    return stt + stamp + exch + gst + sebi + DP_CHARGE

# Indicators
def add_indicators(df):
    df['ema200'] = ta.ema(df['close'], length=200)
    df['ema20']  = ta.ema(df['close'], length=20)
    df['ema50']  = ta.ema(df['close'], length=50)
    adx_df       = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx']    = adx_df['ADX_14'] if 'ADX_14' in adx_df.columns else np.nan
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['donchian_high'] = df['high'].rolling(20).max()
    return df

def willr_signal(df):
    high14 = df['high'].rolling(14).max()
    low14  = df['low'].rolling(14).min()
    df['wpr'] = (high14 - df['close']) / (high14 - low14) * -100
    df['wpr_signal'] = ((df['wpr'] < -80) & (df['wpr'].shift(1) >= -80)).astype(int)
    return df

def breakout_signal(df):
    df['breakout_signal'] = ((df['close'] > df['donchian_high'].shift(1)) &
                             (df['volume'] > VOLUME_MULTIPLIER * df['vol_sma20'])).astype(int)
    return df

def pullback_signal(df):
    pullback = (df['close'] < df['ema20']) & (df['close'] > df['ema50'])
    resume   = (df['close'] > df['ema20']) & (df['volume'] > 1.1 * df['vol_sma20'])
    df['pullback_signal'] = (pullback.shift(1) & resume).astype(int)
    return df

def combine_signals(df):
    df['entry_signal'] = 0
    df['signal_type'] = ""
    df.loc[df['breakout_signal'] == 1, ['entry_signal','signal_type']] = [1,"Breakout"]
    df.loc[(df['entry_signal'] == 0) & (df['pullback_signal'] == 1), ['entry_signal','signal_type']] = [1,"Pullback"]
    df.loc[(df['entry_signal'] == 0) & (df['wpr_signal'] == 1), ['entry_signal','signal_type']] = [1,"W%R"]
    return df

# ML Features
def add_ml_features(df):
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['volumechange'] = df['volume'].pct_change().fillna(0)
    return df

def ml_signals(df, model):
    features = ['rsi','atr','adx','ema10','ema21','volumechange']
    df = add_ml_features(df).dropna(subset=features).reset_index(drop=True)
    df['ml_signal'] = model.predict(df[features])
    return df

def regime_filter(df):
    mask = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD) & (df['volume'] > VOLUME_MULTIPLIER * df['vol_sma20'])
    return df[mask].reset_index(drop=True)

# Backtest core
def backtest(df, symbol):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0

    for i in range(len(df)):
        date = df.at[i,'date']
        close = df.at[i,'close']
        sig = df.at[i,'entry_signal']
        sigtype = df.at[i,'signal_type']

        # exits
        to_close = []
        for pid,pos in positions.items():
            ret = (close - pos['entry_price']) / pos['entry_price']
            if close > pos['high']:
                pos['high'] = close
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = close * (1 - TRAIL_DISTANCE)
            if pos['trail_active']:
                if close * (1 - TRAIL_DISTANCE) > pos['trail_stop']:
                    pos['trail_stop'] = close * (1 - TRAIL_DISTANCE)
            # exit rule
            if ret >= PROFIT_TARGET or ret <= -STOP_LOSS or sig == 0 or (pos['trail_active'] and close <= pos['trail_stop']):
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * close
                charges = calc_charges(buy_val, sell_val)
                pnl = (sell_val - buy_val) - charges
                cash += sell_val
                trades.append([symbol, pos['entry_date'], date, pos['entry_price'], close, (date-pos['entry_date']).days, pnl, ret*100, sigtype])
                to_close.append(pid)
                trade_count += 1
        for pid in to_close:
            positions.pop(pid)

        if trade_count >= MAX_TRADES:
            break
        # entries
        if sig == 1 and len(positions) < MAX_POSITIONS:
            shares = POSITION_SIZE / close
            positions[len(positions)+1] = {'entry_date': date, 'entry_price': close, 'shares': shares, 'high': close, 'trail_active': False, 'trail_stop': 0, 'entry_type': sigtype}

    return trades

# ===== MAIN =====
if __name__ == "__main__":
    model = joblib.load(MODEL_PATH) if USE_ML_CONFIRM else None

    all_trades = []
    cutoff_date = datetime.now() - timedelta(days=5*365)

    for file in os.listdir(DATA_PATH):
        if not file.endswith(".csv"):
            continue
        symbol = file.replace(".csv","")
        df = pd.read_csv(os.path.join(DATA_PATH,file), parse_dates=['date'])
        df = df[df['date'] >= cutoff_date].reset_index(drop=True)
        if df.empty: continue
        
        df = add_indicators(df)
        df = willr_signal(df)
        df = breakout_signal(df)
        df = pullback_signal(df)
        df = combine_signals(df)

        if USE_ML_CONFIRM and model is not None:
            df = ml_signals(df, model)
            df['entry_signal'] = np.where((df['entry_signal']==1)&(df['ml_signal']==1),1,0)

        df = regime_filter(df)
        if df.empty: continue

        trades = backtest(df, symbol)
        all_trades.extend(trades)

    # Save and print summary
    cols = ['symbol','entry_date','exit_date','entry_price','exit_price','days_held','pnl','return_pct','entry_type']
    dftr = pd.DataFrame(all_trades, columns=cols)
    dftr.to_csv(TRADE_LOG_CSV, index=False)
    print(f"Saved trade log for {len(dftr)} trades to {TRADE_LOG_CSV}")

    if not dftr.empty:
        print(dftr.groupby('entry_type')['pnl'].agg(['count','sum','mean']))
        print("\nOverall:")
        print("Total Trades:", len(dftr))
        print("Total PnL:", dftr['pnl'].sum())
        print("Win Rate:", (dftr['pnl']>0).mean()*100, "%")
