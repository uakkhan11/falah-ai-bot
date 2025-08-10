#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta

# ===== CONFIGURATION =====
BASE_DIR = "/root/falah-ai-bot"
DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}
TIMEFRAME = 'daily'  # Choose 'daily', '1hour', or '15minute'
DATA_PATH = DATA_DIRS[TIMEFRAME]

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
RETRAIN_MODEL = False  # Set True to retrain/tune your ML model before testing
USE_ML_CONFIRM = True

TRADE_LOG_CSV = os.path.join(BASE_DIR, f"backtest_trade_log_{TIMEFRAME}.csv")

# Strategy Parameters
INITIAL_CAPITAL   = 1_000_000
POSITION_SIZE     = 100_000
PROFIT_TARGET     = 0.10
STOP_LOSS         = 0.05
TRAIL_TRIGGER     = 0.05
TRAIL_DISTANCE    = 0.02
TRANSACTION_COST  = 0.001
MAX_POSITIONS     = 5
MAX_TRADES        = 2000

# Regime Filters
ADX_THRESHOLD      = 15
VOLUME_MULT        = 1.2

# Bollinger Bands
BB_PERIOD          = 20
BB_STD_DEV         = 2

# CNC cost breakdown
STT_RATE         = 0.001
STAMP_DUTY_RATE  = 0.00015
EXCHANGE_RATE    = 0.0000345
GST_RATE         = 0.18
SEBI_RATE        = 0.000001
DP_CHARGE        = 13.5

# ===== UTILITIES =====
def calc_charges(buy_val, sell_val):
    stt   = (buy_val + sell_val) * STT_RATE
    stamp = buy_val * STAMP_DUTY_RATE
    exch  = (buy_val + sell_val) * EXCHANGE_RATE
    gst   = exch * GST_RATE
    sebi  = (buy_val + sell_val) * SEBI_RATE
    return stt + stamp + exch + gst + sebi + DP_CHARGE

# ===== DATA PREP =====
def load_model():
    if RETRAIN_MODEL:
        # Placeholder: add your retraining code here
        # e.g., model = train_model(new_features, labels)
        # joblib.dump(model, MODEL_PATH)
        pass
    return joblib.load(MODEL_PATH) if USE_ML_CONFIRM else None

def add_indicators(df):
    # Multi-Timeframe Weekly Confirmation
    # Weekly data must be aggregated separately
    if TIMEFRAME == 'daily':
        df_weekly = df.resample('W-MON', on='date').agg({
            'open':'first','high':'max','low':'min','close':'last','volume':'sum'
        }).dropna().reset_index()
        df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20).max()
        df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'].reindex(df['date'], method='ffill').values

    # Daily/Intraday indicators
    df['ema200'] = ta.ema(df['close'], length=200)
    df['adx']    = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['donchian_high'] = df['high'].rolling(20).max()
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=BB_PERIOD, std=BB_STD_DEV)
    if bb is not None:
        # Handle float formatting of std in column names
        upper_col = [c for c in bb.columns if c.startswith(f'BBU_{BB_PERIOD}_')][0]
        lower_col = [c for c in bb.columns if c.startswith(f'BBL_{BB_PERIOD}_')][0]
        df['bb_upper'] = bb[upper_col]
        df['bb_lower'] = bb[lower_col]
# ===== SIGNALS =====
def breakout_signal(df):
    # Daily breakout confirmed by weekly breakout if available
    cond_daily = (df['close'] > df['donchian_high'].shift(1))
    cond_vol   = (df['volume'] > VOLUME_MULT * df['vol_sma20'])
    cond_weekly = True
    if 'weekly_donchian_high' in df.columns:
        cond_weekly = (df['close'] > df['weekly_donchian_high'].shift(1))
    df['breakout_signal'] = (cond_daily & cond_vol & cond_weekly).astype(int)
    return df

def bollinger_breakout_signal(df):
    # Price closes above upper Bollinger Band with volume surge
    df['bb_breakout_signal'] = (
        (df['close'] > df['bb_upper']) &
        (df['volume'] > VOLUME_MULT * df['vol_sma20'])
    ).astype(int)
    return df

def pullback_signal(df):
    # Price pulls to lower BB and then reclaims middle band (mean)
    cond_pull = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    return df

def combine_signals(df):
    # Tag and combine signals in priority order
    df['entry_signal'] = 0
    df['signal_type'] = ''
    # 1. Weekly+Daily breakout
    df.loc[df['breakout_signal']==1, ['entry_signal','signal_type']] = [1,'Breakout']
    # 2. Bollinger Band breakout
    df.loc[(df['entry_signal']==0)&(df['bb_breakout_signal']==1), ['entry_signal','signal_type']] = [1,'BB_Breakout']
    # 3. Bollinger pullback
    df.loc[(df['entry_signal']==0)&(df['bb_pullback_signal']==1), ['entry_signal','signal_type']] = [1,'BB_Pullback']
    return df

# ===== AI FEATURES =====
def add_ml_features(df):
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['volumechange'] = df['volume'].pct_change().fillna(0)
    return df.dropna().reset_index(drop=True)

def ml_filter(df, model):
    features = ['rsi','atr','adx','ema10','ema21','volumechange']
    df = add_ml_features(df)
    df['ml_signal'] = model.predict(df[features])
    # Apply only to breakout signals
    df['entry_signal'] = np.where((df['entry_signal']==1)&(df['ml_signal']==1),1,0)
    df = df[df['entry_signal']==1].reset_index(drop=True)
    return df

# ===== REGIME FILTER =====
def regime_filter(df):
    mask = (
        (df['close'] > df['ema200']) &
        (df['adx'] > ADX_THRESHOLD) &
        (df['volume'] > VOLUME_MULT * df['vol_sma20'])
    )
    return df[mask].reset_index(drop=True)

# ===== BACKTEST ENGINE =====
def backtest(df, symbol):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0

    for i in range(1,len(df)):
        row = df.iloc[i]
        date, price, sig, stype = row['date'], row['close'], row['entry_signal'], row['signal_type']

        # EXIT LOGIC
        for pid,pos in list(positions.items()):
            ret = (price-pos['entry_price'])/pos['entry_price']
            if price>pos['high']: pos['high']=price
            if not pos['trail_active'] and ret>=TRAIL_TRIGGER:
                pos['trail_active']=True
                pos['trail_stop']=price*(1-TRAIL_DISTANCE)
            if pos['trail_active']:
                new_stop = price*(1-TRAIL_DISTANCE)
                if new_stop>pos['trail_stop']: pos['trail_stop']=new_stop
            # Check exit
            if ret>=PROFIT_TARGET or ret<=-STOP_LOSS or sig==0 or (pos['trail_active'] and price<=pos['trail_stop']):
                buy_val = pos['shares']*pos['entry_price']
                sell_val=pos['shares']*price
                charges=calc_charges(buy_val,sell_val)
                exit_val=sell_val*(1-TRANSACTION_COST)
                pnl=exit_val-buy_val-charges
                trades.append({
                    'symbol':symbol,'entry_date':pos['entry_date'],'exit_date':date,
                    'entry_price':pos['entry_price'],'exit_price':price,
                    'days_held':(date-pos['entry_date']).days,'pnl':pnl,
                    'return_pct':ret*100,'entry_type':pos['entry_type'],'exit_reason':(
                        'PT' if ret>=PROFIT_TARGET else
                        'SL' if ret<=-STOP_LOSS else
                        'Signal' if sig==0 else
                        'Trail'
                    )
                })
                cash+=exit_val
                del positions[pid]
                trade_count+=1
                if trade_count>=MAX_TRADES: break
        if trade_count>=MAX_TRADES: break

        # ENTRY LOGIC
        if sig==1 and len(positions)<MAX_POSITIONS and cash>=POSITION_SIZE:
            shares=POSITION_SIZE/price
            positions[len(positions)+1]={
                'entry_date':date,'entry_price':price,
                'shares':shares,'high':price,
                'trail_active':False,'trail_stop':0,
                'entry_type':stype
            }
            cash-=POSITION_SIZE*(1+TRANSACTION_COST)

    return trades

# ===== MAIN EXECUTION =====
if __name__=="__main__":
    model = load_model()
    cutoff_date = datetime.now()-timedelta(days=5*365)
    all_trades=[]

    for file in os.listdir(DATA_PATH):
        if not file.endswith(".csv"): continue
        symbol=file.replace(".csv","")
        df=pd.read_csv(os.path.join(DATA_PATH,file),parse_dates=['date'])
        df=df[df['date']>=cutoff_date].reset_index(drop=True)
        if df.empty: continue

        df=add_indicators(df)
        df=breakout_signal(df)
        df=bollinger_breakout_signal(df)
        df=pullback_signal(df)
        df=combine_signals(df)
        df=regime_filter(df)
        if USE_ML_CONFIRM and model is not None:
            df=ml_filter(df,model)
        if df.empty: continue

        trades=backtest(df,symbol)
        all_trades.extend(trades)

    # Save and summarize
    cols=['symbol','entry_date','exit_date','entry_price','exit_price','days_held','pnl','return_pct','entry_type','exit_reason']
    log_df=pd.DataFrame(all_trades,columns=cols)
    log_df.to_csv(TRADE_LOG_CSV,index=False)
    print(f"Total Trades: {len(log_df)}")
    print("P&L by entry_type:")
    print(log_df.groupby('entry_type')['pnl'].agg(['count','sum','mean']))
    print("Overall PnL:", log_df['pnl'].sum(), "Win Rate:", (log_df['pnl']>0).mean()*100, "%")
