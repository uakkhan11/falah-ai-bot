#!/usr/bin/env python3
import os, pandas as pd, numpy as np
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta

# ==== CONFIG ====
BASE_DIR = "/root/falah-ai-bot"
DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}
TIMEFRAME = 'daily'
DATA_PATH = DATA_DIRS[TIMEFRAME]

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
USE_ML_CONFIRM = True

TRADE_LOG_CSV = os.path.join(BASE_DIR, f"variant_trade_log_{TIMEFRAME}.csv")

# Parameters
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE   = 100_000
PROFIT_TARGET   = 0.10
STOP_LOSS       = 0.05
TRAIL_TRIGGER   = 0.05
TRAIL_DISTANCE  = 0.02
TRANSACTION_COST= 0.001
MAX_POSITIONS   = 5
MAX_TRADES      = 2000

ADX_THRESHOLD   = 15
VOLUME_MULT     = 1.2
BB_PERIOD       = 20
BB_STD_DEV      = 2
CE_ATR_PERIOD   = 22
CE_ATR_MULT     = 3.0
ST_LENGTH       = 10
ST_MULT         = 3.0

# ===== COSTS =====
STT_RATE = 0.001
STAMP_DUTY_RATE=0.00015
EXCHANGE_RATE=0.0000345
GST_RATE=0.18
SEBI_RATE=0.000001
DP_CHARGE=13.5

def calc_charges(buy_val,sell_val):
    stt   = (buy_val+sell_val)*STT_RATE
    stamp = buy_val*STAMP_DUTY_RATE
    exch  = (buy_val+sell_val)*EXCHANGE_RATE
    gst   = exch*GST_RATE
    sebi  = (buy_val+sell_val)*SEBI_RATE
    return stt+stamp+exch+gst+sebi+DP_CHARGE

# ===== INDICATORS =====
def robust_bbcols(bb, period, std):
    std_str = f"{float(std):.1f}"
    upper = [c for c in bb.columns if c.startswith(f"BBU_{period}_")]
    lower = [c for c in bb.columns if c.startswith(f"BBL_{period}_")]
    if not upper: upper = [c for c in bb.columns if 'BBU' in c]
    if not lower: lower = [c for c in bb.columns if 'BBL' in c]
    return upper[0], lower[0]

def add_indicators(df):
    if TIMEFRAME=='daily':
        df_weekly = df.resample('W-MON',on='date').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
        df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20,1).max()
        df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'].reindex(df['date'],method='ffill').values
    df['donchian_high'] = df['high'].rolling(20,1).max()
    df['ema200'] = ta.ema(df['close'],length=200)
    df['adx'] = ta.adx(df['high'],df['low'],df['close'],length=14)['ADX_14']
    df['vol_sma20'] = df['volume'].rolling(20,1).mean()
    bb = ta.bbands(df['close'],length=BB_PERIOD,std=BB_STD_DEV)
    ucol,lcol = robust_bbcols(bb,BB_PERIOD,BB_STD_DEV)
    df['bb_upper'] = bb[ucol]
    df['bb_lower'] = bb[lcol]
    high14 = df['high'].rolling(14).max(); low14 = df['low'].rolling(14).min()
    df['wpr'] = (high14 - df['close'])/(high14-low14)*-100
    atr = ta.atr(df['high'],df['low'],df['close'],length=CE_ATR_PERIOD)
    high20 = df['high'].rolling(CE_ATR_PERIOD,1).max()
    df['chandelier_exit'] = high20 - CE_ATR_MULT * atr
    df['supertrend_dir'] = ta.supertrend(df['high'],df['low'],df['close'],length=ST_LENGTH,multiplier=ST_MULT)[f"SUPERTd_{ST_LENGTH}_{ST_MULT}"]
    return df.dropna().reset_index(drop=True)

# ===== SIGNALS =====
def breakout_signal(df):
    cond_d = df['close']>df['donchian_high'].shift(1)
    cond_v = df['volume']>VOLUME_MULT*df['vol_sma20']
    cond_w = df['close']>df['weekly_donchian_high'].shift(1) if 'weekly_donchian_high' in df else True
    df['breakout_signal'] = (cond_d & cond_v & cond_w).astype(int)
    return df

def bb_breakout_signal(df):
    df['bb_breakout_signal'] = ((df['close']>df['bb_upper']) & (df['volume']>VOLUME_MULT*df['vol_sma20'])).astype(int)
    return df

def bb_pullback_signal(df):
    cond_p = df['close']<df['bb_lower']
    cond_r = df['close']>df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_p.shift(1) & cond_r).astype(int)
    return df

def wpr_signal(df):
    df['wpr_buy_signal'] = ((df['wpr']<-80) & (df['wpr'].shift(1)>=-80)).astype(int)
    return df

def combine_signals(df):
    chand_confirm = df['close']>df['chandelier_exit']
    df['entry_signal'] = 0; df['entry_type']=''
    df.loc[(df['breakout_signal']==1)&chand_confirm,['entry_signal','entry_type']]=[1,'Breakout']
    df.loc[(df['bb_breakout_signal']==1)&chand_confirm&(df['entry_signal']==0),['entry_signal','entry_type']]=[1,'BB_Breakout']
    df.loc[(df['bb_pullback_signal']==1)&chand_confirm&(df['entry_signal']==0),['entry_signal','entry_type']]=[1,'BB_Pullback']
    df.loc[(df['wpr_buy_signal']==1)&chand_confirm&(df['entry_signal']==0),['entry_signal','entry_type']]=[1,'W%R_Buy']
    return df

# ===== AI FILTER =====
def add_ml_features(df):
    df['rsi']=ta.rsi(df['close'],length=14)
    df['atr']=ta.atr(df['high'],df['low'],df['close'],length=14)
    df['ema10']=ta.ema(df['close'],length=10)
    df['ema21']=ta.ema(df['close'],length=21)
    df['volumechange']=df['volume'].pct_change().fillna(0)
    return df.dropna().reset_index(drop=True)

def ml_filter(df, model):
    feats=['rsi','atr','adx','ema10','ema21','volumechange']
    df=add_ml_features(df)
    if df.empty: return df
    X=df[feats]
    if X.shape[0]==0: return df
    df['ml_signal']=model.predict(X)
    df['entry_signal']=np.where((df['entry_signal']==1)&(df['ml_signal']==1),1,0)
    return df[df['entry_signal']==1].reset_index(drop=True)

def regime_filter(df):
    cond=(df['close']>df['ema200'])&(df['adx']>ADX_THRESHOLD)&(df['volume']>VOLUME_MULT*df['vol_sma20'])
    return df[cond].reset_index(drop=True)

# ===== BACKTEST =====
def backtest(df,symbol):
    cash=INITIAL_CAPITAL; positions={}; trades=[]; tc=0
    for i in range(1,len(df)):
        row=df.iloc[i]; date=row['date']; price=row['close']; sig=row['entry_signal']; sigtype=row['entry_type']
        to_close=[]
        for pid,pos in positions.items():
            ret=(price-pos['entry_price'])/pos['entry_price']
            if price>pos['high']: pos['high']=price
            trail_stop=row['chandelier_exit']
            if not pos['trail_active'] and ret>=TRAIL_TRIGGER:
                pos['trail_active']=True; pos['trail_stop']=trail_stop
            if pos['trail_active'] and trail_stop>pos['trail_stop']:
                pos['trail_stop']=trail_stop
            exit_cond = ret>=PROFIT_TARGET or ret<=-STOP_LOSS or sig==0 or (pos['trail_active'] and price<=pos['trail_stop'])
            if exit_cond:
                buyv=pos['shares']*pos['entry_price']
                sellv=pos['shares']*price
                charges=calc_charges(buyv,sellv)
                pnl=sellv*(1-TRANSACTION_COST)-buyv-charges
                if ret>=PROFIT_TARGET: ereas='Profit Target'
                elif ret<=-STOP_LOSS: ereas='Stop Loss'
                elif sig==0: ereas='Signal Exit'
                elif pos['trail_active'] and price<=pos['trail_stop']: ereas='Chandelier Exit'
                else: ereas='Other'
                trades.append({
                    'symbol':symbol,'entry_date':pos['entry_date'],'exit_date':date,
                    'entry_price':pos['entry_price'],'exit_price':price,
                    'days_held':(date-pos['entry_date']).days,'pnl':pnl,'return_pct':ret*100,
                    'entry_type':pos['entry_type'],'exit_reason':ereas
                })
                cash+=sellv; to_close.append(pid); tc+=1
                if tc>=MAX_TRADES: break
        for pid in to_close: positions.pop(pid)
        if tc>=MAX_TRADES: break
        if sig==1 and len(positions)<MAX_POSITIONS and cash>=POSITION_SIZE:
            shares=POSITION_SIZE/price
            positions[len(positions)+1]={'entry_date':date,'entry_price':price,'shares':shares,'high':price,'trail_active':False,'trail_stop':0,'entry_type':sigtype}
            cash-=POSITION_SIZE*(1+TRANSACTION_COST)
    return trades

# ===== MAIN =====
if __name__=="__main__":
    model=joblib.load(MODEL_PATH) if USE_ML_CONFIRM else None
    cutoff=datetime.now()-timedelta(days=5*365)
    all_tr=[]
    for file in os.listdir(DATA_PATH):
        if not file.endswith(".csv"): continue
        df=pd.read_csv(os.path.join(DATA_PATH,file),parse_dates=['date'])
        df=df[df['date']>=cutoff].reset_index(drop=True)
        df=add_indicators(df)
        df=breakout_signal(df); df=bb_breakout_signal(df); df=bb_pullback_signal(df); df=wpr_signal(df)
        df=combine_signals(df); df=regime_filter(df)
        if USE_ML_CONFIRM and model is not None:
            df=ml_filter(df,model)
        if df.empty: continue
        all_tr.extend(backtest(df,file.replace(".csv","")))
    log_df=pd.DataFrame(all_tr); log_df.to_csv(TRADE_LOG_CSV,index=False)
    print(f"Total Trades: {len(log_df)}")
    print("P&L by entry_type:"); print(log_df.groupby('entry_type')['pnl'].agg(['count','sum','mean']))
    print("Overall PnL:", log_df['pnl'].sum(),"Win Rate:",(log_df['pnl']>0).mean()*100,"%")
    print("\nExit reason performance:"); print(log_df.groupby('exit_reason')['pnl'].agg(['count','sum','mean']).sort_values('sum',ascending=False))
    # --- Extra reports ---
    print("\nPer-symbol performance:\n", log_df.groupby('symbol')['pnl'].agg(['count','sum','mean']).sort_values('sum',ascending=False))
    log_df['exit_date']=pd.to_datetime(log_df['exit_date']); print("\nYearly trade breakdown:\n", log_df.groupby(log_df['exit_date'].dt.year)['pnl'].agg(['count','sum','mean']))
    # Metrics
    log_df=log_df.sort_values('exit_date'); log_df['cum_pnl']=log_df['pnl'].cumsum()
    roll_max=log_df['cum_pnl'].cummax(); dd=roll_max-log_df['cum_pnl']; max_dd=dd.max()
    sharpe=log_df['return_pct'].mean()/log_df['return_pct'].std() if log_df['return_pct'].std()!=0 else np.nan
    print(f"\nMax Drawdown: {max_dd:.2f} | Sharpe: {sharpe:.2f} | Avg Hold Days: {log_df['days_held'].mean():.2f}")
