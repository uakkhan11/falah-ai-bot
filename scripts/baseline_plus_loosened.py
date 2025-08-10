#!/usr/bin/env python3
import os, pandas as pd, numpy as np
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta

pd.set_option('future.no_silent_downcasting', True)

# ==== CONFIG ====
BASE_DIR = "/root/falah-ai-bot"
DATA_DIR = os.path.join(BASE_DIR, "swing_data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
USE_ML_CONFIRM = True

INITIAL_CAPITAL = 1_000_000
POSITION_SIZE   = 100_000
PROFIT_TARGET   = 0.10
ATR_SL_MULT     = 2.0
ATR_PERIOD      = 14
TRAIL_TRIGGER   = 0.07
TRAIL_DISTANCE  = 0.03
TRANSACTION_COST= 0.001
MAX_POSITIONS   = 5
MAX_TRADES      = 2000

ADX_THRESHOLD   = 15
VOLUME_MULT     = 1.2

# Cost structure
STT_RATE=0.001; STAMP_DUTY_RATE=0.00015; EXCHANGE_RATE=0.0000345; GST_RATE=0.18; SEBI_RATE=0.000001; DP_CHARGE=13.5

def calc_charges(buy_val, sell_val):
    stt   = (buy_val+sell_val)*STT_RATE
    stamp = buy_val*STAMP_DUTY_RATE
    exch  = (buy_val+sell_val)*EXCHANGE_RATE
    gst   = exch*GST_RATE
    sebi  = (buy_val+sell_val)*SEBI_RATE
    return stt+stamp+exch+gst+sebi+DP_CHARGE

# ---- Indicators ----
def robust_bbcols(bb):
    ucol = [c for c in bb.columns if 'BBU' in c][0]
    lcol = [c for c in bb.columns if 'BBL' in c][0]
    return ucol, lcol

def add_indicators(df):
    if 'date' in df.columns:
        df = df.sort_values('date')
    # Weekly Donchian
    df_weekly = df.resample('W-MON', on='date').agg({
        'open':'first','high':'max','low':'min','close':'last','volume':'sum'
    }).dropna().reset_index()
    df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20,1).max()
    df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high']\
        .reindex(df['date'], method='ffill').values

    df['donchian_high'] = df['high'].rolling(20,1).max()
    df['ema200'] = ta.ema(df['close'], length=200)
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df else np.nan
    except: df['adx'] = np.nan
    df['vol_sma20'] = df['volume'].rolling(20,1).mean()

    bb = ta.bbands(df['close'], length=20, std=2)
    ucol, lcol = robust_bbcols(bb)
    df['bb_upper'] = bb[ucol]; df['bb_lower'] = bb[lcol]

    high14 = df['high'].rolling(14).max()
    low14  = df['low'].rolling(14).min()
    df['wpr'] = (high14 - df['close'])/(high14-low14) * -100

    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    atr_ce = ta.atr(df['high'], df['low'], df['close'], length=22)
    high20 = df['high'].rolling(22,1).max()
    df['chandelier_exit'] = high20 - 3.0 * atr_ce

    try:
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['supertrend'] = st['SUPERT_10_3.0']
        df['supertrend_dir'] = st['SUPERTd_10_3.0']
    except:
        df['supertrend'] = np.nan; df['supertrend_dir']=0

    cols = ['close','donchian_high','ema200','adx','vol_sma20','bb_upper','bb_lower','wpr','atr','chandelier_exit','supertrend','supertrend_dir']
    df[cols] = df[cols].replace({None: np.nan}).infer_objects(copy=False)
    return df.reset_index(drop=True)

# ---- Signals (loosening rules) ----
def breakout_signal(df):
    cond_d = df['close'] > df['donchian_high'].shift(1)
    cond_v = df['volume'] > VOLUME_MULT * df['vol_sma20']
    cond_w = df['close'] > df['weekly_donchian_high'].shift(1)
    df['breakout_signal'] = (cond_d & cond_v & cond_w).astype(int)
    return df

def bb_breakout_signal(df):
    df['bb_breakout_signal'] = ((df['close'] > df['bb_upper']) & (df['volume'] > VOLUME_MULT * df['vol_sma20'])).astype(int)
    return df

def bb_pullback_signal(df):
    cond_pull = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    return df

def combine_signals(df):
    regime = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD)
    chand_or_st = (df['close'] > df['chandelier_exit']) | (df['supertrend_dir'] == 1)
    mask = regime & chand_or_st

    df['entry_signal'] = 0; df['entry_type'] = ''
    df.loc[(df['breakout_signal']==1) & mask, ['entry_signal','entry_type']] = [1,'Breakout']
    df.loc[(df['bb_breakout_signal']==1) & mask & (df['entry_signal']==0), ['entry_signal','entry_type']] = [1,'BB_Breakout']
    df.loc[(df['bb_pullback_signal']==1) & mask & (df['entry_signal']==0), ['entry_signal','entry_type']] = [1,'BB_Pullback']
    return df

# ---- AI Filter ----
def apply_ml_filter(df, model):
    if not USE_ML_CONFIRM or model is None:
        df['ml_signal'] = 1
        return df
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['volumechange'] = df['volume'].pct_change().fillna(0)
    features = ['rsi','atr','adx','ema10','ema21','volumechange']
    df = df.dropna(subset=features).reset_index(drop=True)
    if df.empty: return df
    df['ml_signal'] = model.predict(df[features])
    df['entry_signal'] = np.where((df['entry_signal']==1) & (df['ml_signal']==1), 1, 0)
    return df

# ---- Backtest ----
def backtest(df, symbol):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0
    for i in range(1,len(df)):
        row = df.iloc[i]
        date, price, sig, sigtype = row['date'], row['close'], row['entry_signal'], row['entry_type']

        # EXIT
        to_close = []
        for pid,pos in positions.items():
            ret = (price - pos['entry_price']) / pos['entry_price']
            atr_stop = pos['entry_price'] - ATR_SL_MULT * pos['entry_atr']

            # Trailing via CE
            if price > pos['high']: pos['high'] = price
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = row['chandelier_exit']
            if pos['trail_active'] and row['chandelier_exit'] > pos['trail_stop']:
                pos['trail_stop'] = row['chandelier_exit']

            if ret >= PROFIT_TARGET: reason = 'Profit Target'
            elif price <= atr_stop: reason = 'ATR Stop Loss'
            elif pos['trail_active'] and price <= pos['trail_stop']: reason = 'Chandelier Exit'
            elif (row['close'] < row['ema200']) or (row['adx'] <= ADX_THRESHOLD): reason = 'Regime Exit'
            else: reason = None

            if reason:
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                charges = calc_charges(buy_val, sell_val)
                pnl = sell_val*(1-TRANSACTION_COST) - buy_val - charges
                trades.append({'symbol':symbol,'entry_date':pos['entry_date'],'exit_date':date,'pnl':pnl,'entry_type':pos['entry_type'],'exit_reason':reason})
                cash += sell_val; to_close.append(pid); trade_count += 1
                if trade_count >= MAX_TRADES: break
        for pid in to_close: positions.pop(pid)
        if trade_count >= MAX_TRADES: break

        # ENTRY
        if sig == 1 and len(positions) < MAX_POSITIONS and cash >= POSITION_SIZE:
            shares = POSITION_SIZE / price
            positions[len(positions)+1] = {'entry_date':date,'entry_price':price,'shares':shares,'high':price,'trail_active':False,'trail_stop':0,'entry_atr':row['atr'],'entry_type':sigtype}
            cash -= POSITION_SIZE * (1 + TRANSACTION_COST)
    return trades

# ---- Main ----
if __name__ == "__main__":
    model = joblib.load(MODEL_PATH) if USE_ML_CONFIRM and os.path.exists(MODEL_PATH) else None
    all_trades = []
    cutoff = datetime.now() - timedelta(days=5*365)

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"): continue
        symbol = file.replace(".csv","")
        df = pd.read_csv(os.path.join(DATA_DIR,file), parse_dates=['date'])
        df = df[df['date'] >= cutoff].reset_index(drop=True)
        if df.empty: continue

        df = add_indicators(df)
        df = breakout_signal(df)
        df = bb_breakout_signal(df)
        df = bb_pullback_signal(df)
        df = combine_signals(df)
        df = apply_ml_filter(df, model)
        if df.empty: continue

        trades = backtest(df, symbol)
        all_trades.extend(trades)

    log_df = pd.DataFrame(all_trades)
    log_df.to_csv("baseline_plus_loosened_log.csv", index=False)
    print(f"Total Trades: {len(log_df)} | Overall PnL: {log_df['pnl'].sum():,.0f} | Win Rate: {(log_df['pnl']>0).mean()*100:.2f}%")
    print("\nPnL by entry_type:\n", log_df.groupby('entry_type')['pnl'].agg(['count','sum','mean']))
    print("\nExit reason performance:\n", log_df.groupby('exit_reason')['pnl'].agg(['count','sum','mean']))
