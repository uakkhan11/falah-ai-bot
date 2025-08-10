#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
baseline_plus_v3_wfo_fast.py
Same trading logic as v3, but with indicator caching and parallel parameter search for much faster execution.
"""

import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta
import itertools
from joblib import Parallel, delayed

pd.set_option('future.no_silent_downcasting', True)

# ==== CONFIG ====
BASE_DIR   = "/root/falah-ai-bot"
DATA_DIR   = os.path.join(BASE_DIR, "swing_data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

USE_ML = True
N_JOBS  = -1   # All CPU cores

# General params
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE   = 100_000
PROFIT_TARGET   = 0.10
ATR_PERIOD      = 14
TRAIL_TRIGGER   = 0.07
TRANSACTION_COST= 0.001
MAX_POSITIONS   = 5
MAX_TRADES      = 2000
ADX_THRESHOLD_PULL = 15
VOLUME_PULL     = 1.2

# Cost rates
STT_RATE=0.001; STAMP_DUTY_RATE=0.00015; EXCHANGE_RATE=0.0000345
GST_RATE=0.18;  SEBI_RATE=0.000001; DP_CHARGE=13.5

# WFO parameter grid
PARAM_GRID = {
    'ATR_SL_BREAK': [2.5, 3.0],
    'ATR_SL_PULL': [3.0, 3.5],
    'ADX_BREAK': [8, 10, 12],
    'VOL_BREAK': [1.05, 1.1],
    'REG_FAIL_BARS': [2, 3]
}

# ==== Helper ====
def calc_charges(buy_val,sell_val):
    stt   = (buy_val+sell_val)*STT_RATE
    stamp = buy_val*STAMP_DUTY_RATE
    exch  = (buy_val+sell_val)*EXCHANGE_RATE
    gst   = exch*GST_RATE
    sebi  = (buy_val+sell_val)*SEBI_RATE
    return stt+stamp+exch+gst+sebi+DP_CHARGE

def robust_bbcols(bb):
    if bb is None or not hasattr(bb, 'columns'):
        return None, None
    upper = [c for c in bb.columns if 'BBU' in c]
    lower = [c for c in bb.columns if 'BBL' in c]
    return (upper[0] if upper else None, lower[0] if lower else None)

# ==== Indicators ====
def add_indicators(df):
    if 'date' in df.columns:
        df = df.sort_values('date')
        # Weekly Donchian
    df_weekly = df.resample('W-MON', on='date').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna().reset_index()
    df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20, min_periods=1).max()
    df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high']\
        .reindex(df['date'], method='ffill').values

    df['donchian_high'] = df['high'].rolling(20, min_periods=1).max()
    df['ema200'] = ta.ema(df['close'], length=200)

    # ADX
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None else np.nan
    except:
        df['adx'] = np.nan

    # Volume MA
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()

    # Bollinger Bands
    try:
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
            ucol = [c for c in bb.columns if 'BBU' in c][0]
            lcol = [c for c in bb.columns if 'BBL' in c][0]
            df['bb_upper'] = bb[ucol]
            df['bb_lower'] = bb[lcol]
        else:
            df['bb_upper'] = np.nan
            df['bb_lower'] = np.nan
    except:
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan

    # W%R
    high14 = df['high'].rolling(14, min_periods=1).max()
    low14  = df['low'].rolling(14, min_periods=1).min()
    df['wpr'] = (high14 - df['close']) / (high14 - low14) * -100

    # ATR
    try:
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    except:
        df['atr'] = np.nan

    # Chandelier Exit
    try:
        atr_ce = ta.atr(df['high'], df['low'], df['close'], length=22)
        high20 = df['high'].rolling(22, min_periods=1).max()
        if atr_ce is not None:
            df['chandelier_exit'] = high20 - 3.0 * atr_ce
        else:
            df['chandelier_exit'] = np.nan
    except:
        df['chandelier_exit'] = np.nan

    # === Ensure RSI is always present ===
    try:
        df['rsi'] = ta.rsi(df['close'], length=14)
    except:
        df['rsi'] = np.nan

    # Fill RSI NaNs forward if possible, else leave — will be handled later
       df['rsi'] = df['rsi'].ffill().bfill()

    return df.replace({None: np.nan}).infer_objects(copy=False).reset_index(drop=True)

# ==== Signals ====
def breakout_signal(df, vol_mult):
    cond = (df['close'] > df['donchian_high'].shift(1)) & \
           (df['volume'] > vol_mult * df['vol_sma20']) & \
           (df['close'] > df['weekly_donchian_high'].shift(1))
    df['breakout_signal'] = cond.astype(int)
    return df

def bb_breakout_signal(df, vol_mult):
    cond = (df['close'] > df['bb_upper']) & \
           (df['volume'] > vol_mult * df['vol_sma20'])
    df['bb_breakout_signal'] = cond.astype(int)
    return df

def bb_pullback_signal(df):
    cond_pull = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    return df

def combine_signals(df, params):
    chand_or_st = (df['close'] > df['chandelier_exit'])  # v3 doesn't use ST here
    reg_break = (df['close'] > df['ema200']) & (df['adx'] > params['ADX_BREAK'])
    reg_pull  = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD_PULL)
    df['entry_signal'], df['entry_type'] = 0, ''
    df.loc[(df['breakout_signal']==1) & chand_or_st & reg_break, ['entry_signal','entry_type']] = [1,'Breakout']
    df.loc[(df['bb_breakout_signal']==1) & chand_or_st & reg_break & (df['entry_signal']==0), ['entry_signal','entry_type']] = [1,'BB_Breakout']
    df.loc[(df['bb_pullback_signal']==1) & chand_or_st & reg_pull & (df['entry_signal']==0), ['entry_signal','entry_type']] = [1,'BB_Pullback']
    return df

# ==== ML filter ====
def apply_ml_filter(df, model):
    if not USE_ML or model is None: return df
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['volumechange'] = df['volume'].pct_change().fillna(0)
    features = ['atr','adx','ema10','ema21','volumechange']
    df = df.dropna(subset=features).reset_index(drop=True)
    df['ml_signal'] = model.predict(df[features])
    df['entry_signal'] = np.where((df['entry_signal']==1) & (df['ml_signal']==1), 1, 0)
    return df

# ==== Backtest ====
def backtest(df, symbol, params):
    cash = INITIAL_CAPITAL; positions = {}; trades = []; trade_count = 0
    fail_count = {}
    for i in range(1,len(df)):
        row = df.iloc[i]
        date, price, sig = row['date'], row['close'], row['entry_signal']
        sigtype = row['entry_type']
        regime_ok = (row['close'] > row['ema200']) and (row['adx'] > ADX_THRESHOLD_PULL)

        to_close=[]
        for pid,pos in positions.items():
            ret = (price - pos['entry_price']) / pos['entry_price']
            atr_stop = pos['entry_price'] - (params['ATR_SL_BREAK'] if pos['entry_type'] in ['Breakout','BB_Breakout'] else params['ATR_SL_PULL']) * pos['entry_atr']
            if (date - pos['entry_date']).days < 3: atr_stop = -np.inf
            if price > pos['high']: pos['high'] = price
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True; pos['trail_stop'] = row['chandelier_exit']
            if pos['trail_active'] and row['chandelier_exit'] > pos['trail_stop']:
                pos['trail_stop'] = row['chandelier_exit']
            if ret >= PROFIT_TARGET: reason = 'Profit Target'
            elif price <= atr_stop: reason = 'ATR Stop Loss'
            elif pos['trail_active'] and price <= pos['trail_stop']: reason = 'Chandelier Exit'
            else:
                key=(symbol,pid)
                if not regime_ok:
                    fail_count[key] = fail_count.get(key,0)+1
                else:
                    fail_count[key]=0
                reason = 'Regime Exit' if fail_count.get(key,0) >= params['REG_FAIL_BARS'] else None
            if reason:
                buy_val = pos['shares']*pos['entry_price']
                sell_val= pos['shares']*price
                pnl = sell_val*(1-TRANSACTION_COST)-buy_val-calc_charges(buy_val,sell_val)
                trades.append({'symbol':symbol,'pnl':pnl,'entry_type':pos['entry_type'],'exit_reason':reason})
                cash += sell_val; to_close.append(pid); trade_count+=1
        for pid in to_close: positions.pop(pid)
        if sig==1 and len(positions)<MAX_POSITIONS and cash>=POSITION_SIZE:
            shares = POSITION_SIZE / price
            positions[len(positions)+1] = {
                'entry_date':date,'entry_price':price,'shares':shares,'high':price,
                'trail_active':False,'trail_stop':0,
                'entry_atr':row['atr'],'entry_type':sigtype
            }
            cash -= POSITION_SIZE*(1+TRANSACTION_COST)
    return trades

# ==== Parallelised WFO ====
def walk_forward(all_data, train_years=2, test_months=6):
    model = joblib.load(MODEL_PATH) if USE_ML and os.path.exists(MODEL_PATH) else None
    current = all_data['date'].min(); end_date = all_data['date'].max()
    results=[]

    while current < end_date:
        train_end = current + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_end  = train_end + pd.DateOffset(months=test_months)
        train_df = all_data[(all_data['date']>=current) & (all_data['date']<=train_end)]
        test_df  = all_data[(all_data['date']>train_end) & (all_data['date']<=test_end)]
        if train_df.empty or test_df.empty: break

        # === Step 1: Cache indicators for this window ===
        train_cache = {sym: add_indicators(train_df[train_df['symbol']==sym]) for sym in train_df['symbol'].unique()}
        test_cache  = {sym: add_indicators(test_df[test_df['symbol']==sym])  for sym in test_df['symbol'].unique()}

        # === Step 2: Parameter optimisation in parallel ===
        def eval_params(params):
            trades=[]
            for sym, df in train_cache.items():
                dfp = breakout_signal(df.copy(), params['VOL_BREAK'])
                dfp = bb_breakout_signal(dfp, params['VOL_BREAK'])
                dfp = bb_pullback_signal(dfp)
                dfp = combine_signals(dfp, params)
                dfp = apply_ml_filter(dfp, model)
                if not dfp.empty:
                    trades += backtest(dfp, sym, params)
            if trades:
                tdf=pd.DataFrame(trades)
                return params, tdf['pnl'].sum()*(tdf['pnl']>0).mean()
            return params, -np.inf

        param_sets = [dict(zip(PARAM_GRID.keys(), comb)) for comb in itertools.product(*PARAM_GRID.values())]
        scores = Parallel(n_jobs=N_JOBS)(delayed(eval_params)(p) for p in param_sets)
        best_params = max(scores, key=lambda x: x[1])[0]

        # === Step 3: Test with best params ===
        for sym, df in test_cache.items():
            dfp = breakout_signal(df.copy(), best_params['VOL_BREAK'])
            dfp = bb_breakout_signal(dfp, best_params['VOL_BREAK'])
            dfp = bb_pullback_signal(dfp)
            dfp = combine_signals(dfp, best_params)
            dfp = apply_ml_filter(dfp, model)
            if not dfp.empty:
                results += backtest(dfp, sym, best_params)

        current = test_end

    return pd.DataFrame(results)

# ==== Main ====
if __name__=="__main__":
    all_data=[]
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            sym=file.replace(".csv","")
            df=pd.read_csv(os.path.join(DATA_DIR,file),parse_dates=['date'])
            df=df[df['date']>datetime.now()-timedelta(days=5*365)]
            df['symbol']=sym
            all_data.append(df)
    all_df=pd.concat(all_data,ignore_index=True)
    log_df = walk_forward(all_df)
    log_df.to_csv("v3_fast_trades.csv",index=False)
    print(f"Total Trades: {len(log_df)} | PnL: {log_df['pnl'].sum():,.0f} | Win Rate: {(log_df['pnl']>0).mean()*100:.2f}%")
