#!/usr/bin/env python3
import os, pandas as pd, numpy as np
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta
import itertools

pd.set_option('future.no_silent_downcasting', True)

# ==== CONFIG ====
BASE_DIR     = "/root/falah-ai-bot"
DATA_DIR     = os.path.join(BASE_DIR, "swing_data")
MODEL_PATH   = os.path.join(BASE_DIR, "model.pkl")
USE_ML       = True
WFO_MODE     = True   # Set False to just run fixed params
FIXED_PARAMS = { # used if WFO_MODE=False
    'ATR_SL_BREAK': 2.5,
    'ATR_SL_PULL': 3.0,
    'ADX_BREAK': 10,
    'VOL_BREAK': 1.05,
    'REG_FAIL_BARS': 3
}

# general
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

# costs (India example)
STT_RATE=0.001; STAMP_DUTY_RATE=0.00015; EXCHANGE_RATE=0.0000345; GST_RATE=0.18; SEBI_RATE=0.000001; DP_CHARGE=13.5

# ==== PARAMETER GRID for WFO ====
PARAM_GRID = {
    'ATR_SL_BREAK': [2.5, 3.0],
    'ATR_SL_PULL': [3.0, 3.5],
    'ADX_BREAK': [8, 10, 12],
    'VOL_BREAK': [1.05, 1.1],
    'REG_FAIL_BARS': [2, 3]
}

# ==== FUNCTIONS ====
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
    u_candidates = [c for c in bb.columns if 'BBU' in c]
    l_candidates = [c for c in bb.columns if 'BBL' in c]
    ucol = u_candidates[0] if u_candidates else None
    lcol = l_candidates[0] if l_candidates else None
    return ucol, lcol

def add_indicators(df):
    if 'date' in df.columns: df = df.sort_values('date')
    df_weekly = df.resample('W-MON', on='date').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
    df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20,1).max()
    df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'].reindex(df['date'], method='ffill').values

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
        df['supertrend_dir'] = st['SUPERTd_10_3.0']
    except: df['supertrend_dir'] = 0

    return df.replace({None: np.nan}).infer_objects(copy=False).reset_index(drop=True)

def breakout_signal(df, vol_mult):
    cond_d = df['close'] > df['donchian_high'].shift(1)
    cond_v = df['volume'] > vol_mult * df['vol_sma20']
    cond_w = df['close'] > df['weekly_donchian_high'].shift(1)
    df['breakout_signal'] = (cond_d & cond_v & cond_w).astype(int)
    return df

def bb_breakout_signal(df, vol_mult):
    df['bb_breakout_signal'] = ((df['close'] > df['bb_upper']) &
                                (df['volume'] > vol_mult * df['vol_sma20'])).astype(int)
    return df

def bb_pullback_signal(df):
    cond_pull   = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    return df

def combine_signals(df, params):
    chand_or_st = (df['close'] > df['chandelier_exit']) | (df['supertrend_dir']==1)
    reg_break   = (df['close'] > df['ema200']) & (df['adx'] > params['ADX_BREAK'])
    reg_pull    = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD_PULL)
    df['entry_signal'] = 0; df['entry_type'] = ''
    df.loc[(df['breakout_signal']==1) & chand_or_st & reg_break, ['entry_signal','entry_type']] = [1,'Breakout']
    df.loc[(df['bb_breakout_signal']==1) & chand_or_st & reg_break & (df['entry_signal']==0), ['entry_signal','entry_type']] = [1,'BB_Breakout']
    df.loc[(df['bb_pullback_signal']==1) & chand_or_st & reg_pull & (df['entry_signal']==0), ['entry_signal','entry_type']] = [1,'BB_Pullback']
    return df

def apply_ml_filter(df, model):
    if not USE_ML or model is None: return df
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

def backtest(df, symbol, params):
    cash=INITIAL_CAPITAL; positions={}; trades=[]; trade_count=0; fail_count={}
    for i in range(1,len(df)):
        row = df.iloc[i]
        date, price, sig, sigtype = row['date'], row['close'], row['entry_signal'], row['entry_type']
        regime_ok = (row['close'] > row['ema200']) and (row['adx'] > ADX_THRESHOLD_PULL)
        st_up     = row['supertrend_dir'] == 1
        to_close=[]
        for pid,pos in positions.items():
            ret = (price - pos['entry_price']) / pos['entry_price']
            # ATR SL dynamic
            if pos['entry_type'] in ['Breakout','BB_Breakout']:
                atr_stop = pos['entry_price'] - params['ATR_SL_BREAK'] * pos['entry_atr']
            else:
                atr_stop = pos['entry_price'] - params['ATR_SL_PULL'] * pos['entry_atr']
            # Delay ATR SL until 3 bars
            if (date - pos['entry_date']).days < 3:
                atr_stop = -np.inf
            # trailing
            if price > pos['high']: pos['high']=price
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active']=True; pos['trail_stop']=row['chandelier_exit']
            if pos['trail_active'] and row['chandelier_exit'] > pos['trail_stop']:
                pos['trail_stop']=row['chandelier_exit']
            # exit conditions
            if ret >= PROFIT_TARGET: reason = 'Profit Target'
            elif price <= atr_stop: reason = 'ATR Stop Loss'
            elif pos['trail_active'] and price <= pos['trail_stop']: reason = 'Chandelier Exit'
            else:
                key=f"{symbol}_{pid}"
                if not (regime_ok and st_up):
                    fail_count[key] = fail_count.get(key,0) + 1
                else: fail_count[key]=0
                reason = 'Regime Exit' if fail_count.get(key,0) >= params['REG_FAIL_BARS'] else None
            if reason:
                buy_val = pos['shares']*pos['entry_price']
                sell_val= pos['shares']*price
                pnl = sell_val*(1-TRANSACTION_COST)-buy_val-calc_charges(buy_val,sell_val)
                trades.append({'symbol':symbol,'entry_date':pos['entry_date'],'exit_date':date,
                               'pnl':pnl,'entry_type':pos['entry_type'],'exit_reason':reason,
                               'days_held':(date-pos['entry_date']).days,'atr_entry':pos['entry_atr']})
                cash+=sell_val; to_close.append(pid); trade_count+=1
        for pid in to_close: positions.pop(pid)
        if sig==1 and len(positions)<MAX_POSITIONS and cash>=POSITION_SIZE:
            shares = POSITION_SIZE/price
            positions[len(positions)+1] = {'entry_date':date,'entry_price':price,
                                           'shares':shares,'high':price,
                                           'trail_active':False,'trail_stop':0,
                                           'entry_atr':row['atr'],'entry_type':sigtype}
            cash -= POSITION_SIZE*(1+TRANSACTION_COST)
    return trades

# ==== WALK-FORWARD ====
def walk_forward(all_data, train_years=2, test_months=6):
    start_year = all_data['date'].dt.year.min()
    end_year   = all_data['date'].dt.year.max()
    model = joblib.load(MODEL_PATH) if USE_ML and os.path.exists(MODEL_PATH) else None
    current = all_data['date'].min()
    results=[]
    while current < all_data['date'].max():
        train_end   = current + pd.DateOffset(years=train_years) - pd.DateOffset(days=1)
        test_end    = train_end + pd.DateOffset(months=test_months)
        train_df = all_data[(all_data['date']>=current) & (all_data['date']<=train_end)]
        test_df  = all_data[(all_data['date']>train_end) & (all_data['date']<=test_end)]
        if train_df.empty or test_df.empty: break
        # optimise
        best_params=None; best_score=-np.inf
        for comb in itertools.product(*PARAM_GRID.values()):
            params = dict(zip(PARAM_GRID.keys(), comb))
            trades=[]
            for sym in train_df['symbol'].unique():
                df = train_df[train_df['symbol']==sym].copy()
                df = add_indicators(df)
                df = breakout_signal(df, params['VOL_BREAK'])
                df = bb_breakout_signal(df, params['VOL_BREAK'])
                df = bb_pullback_signal(df)
                df = combine_signals(df, params)
                df = apply_ml_filter(df, model)
                if not df.empty:
                    trades += backtest(df,sym,params)
            if trades:
                tdf=pd.DataFrame(trades); score = tdf['pnl'].sum()*(tdf['pnl']>0).mean()
                if score>best_score: best_score, best_params=score, params
        # test best params
        test_trades=[]
        for sym in test_df['symbol'].unique():
            df = test_df[test_df['symbol']==sym].copy()
            df = add_indicators(df)
            df = breakout_signal(df, best_params['VOL_BREAK'])
            df = bb_breakout_signal(df, best_params['VOL_BREAK'])
            df = bb_pullback_signal(df)
            df = combine_signals(df, best_params)
            df = apply_ml_filter(df, model)
            if not df.empty:
                test_trades += backtest(df,sym,best_params)
        results += test_trades
        current = test_end
    return pd.DataFrame(results)

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
    if WFO_MODE:
        log_df = walk_forward(all_df)
    else:
        params = FIXED_PARAMS
        model = joblib.load(MODEL_PATH) if USE_ML and os.path.exists(MODEL_PATH) else None
        trades=[]
        for sym in all_df['symbol'].unique():
            df = all_df[all_df['symbol']==sym].copy()
            df = add_indicators(df)
            df = breakout_signal(df, params['VOL_BREAK'])
            df = bb_breakout_signal(df, params['VOL_BREAK'])
            df = bb_pullback_signal(df)
            df = combine_signals(df, params)
            df = apply_ml_filter(df, model)
            if not df.empty:
                trades += backtest(df,sym,params)
        log_df=pd.DataFrame(trades)
    log_df.to_csv("v3_wfo_trades.csv",index=False)
    print(f"Total Trades: {len(log_df)} | PnL: {log_df['pnl'].sum():,.0f} | Win Rate: {(log_df['pnl']>0).mean()*100:.2f}%")
    print("\nBy entry_type:\n",log_df.groupby('entry_type')['pnl'].agg(['count','sum','mean']))
    print("\nBy exit_reason:\n",log_df.groupby('exit_reason')['pnl'].agg(['count','sum','mean']))
