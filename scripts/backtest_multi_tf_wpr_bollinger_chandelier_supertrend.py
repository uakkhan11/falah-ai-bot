#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta

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

TRADE_LOG_CSV = os.path.join(BASE_DIR, f"backtest_trade_log_{TIMEFRAME}.csv")

INITIAL_CAPITAL   = 1_000_000
POSITION_SIZE     = 100_000
PROFIT_TARGET     = 0.10
STOP_LOSS         = 0.05
TRAIL_TRIGGER     = 0.05
TRAIL_DISTANCE    = 0.02
TRANSACTION_COST  = 0.001
MAX_POSITIONS     = 5
MAX_TRADES        = 2000

ADX_THRESHOLD      = 15
VOLUME_MULT        = 1.2

BB_PERIOD          = 20
BB_STD_DEV         = 2
CE_ATR_PERIOD      = 22
CE_ATR_MULTIPLIER  = 3.0
ST_LENGTH          = 10
ST_MULTIPLIER      = 3.0

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

def robust_bbcols(bb, period, std):
    upper = None
    lower = None
    std_str = f"{float(std):.1f}"
    for col in bb.columns:
        if col.startswith(f"BBU_{period}_"):
            upper = col
        if col.startswith(f"BBL_{period}_"):
            lower = col            
    if upper is None or lower is None:
        for col in bb.columns:
            if col.startswith(f'BBU_{period}_{std_str}'):
                upper = col
            if col.startswith(f'BBL_{period}_{std_str}'):
                lower = col
    if upper is None:
        upper = [c for c in bb.columns if 'BBU' in c][0]
    if lower is None:
        lower = [c for c in bb.columns if 'BBL' in c][0]
    return upper, lower

def load_model():
    if USE_ML_CONFIRM and os.path.isfile(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def add_indicators(df):
    if TIMEFRAME == 'daily':
        df_weekly = df.resample('W-MON', on='date').agg({
            'open':'first','high':'max','low':'min','close':'last','volume':'sum'
        }).dropna().reset_index()
        df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20, min_periods=1).max()
        df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'].reindex(df['date'], method='ffill').values

    if 'high' in df.columns and len(df) >= 20:
        df['donchian_high'] = df['high'].rolling(20, min_periods=1).max()
    else:
        df['donchian_high'] = np.nan

    df['ema200'] = ta.ema(df['close'], length=200)
    df['adx'] = np.nan
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx_df is not None and 'ADX_14' in adx_df.columns:
        df['adx'] = adx_df['ADX_14']

    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()

    bb = ta.bbands(df['close'], length=BB_PERIOD, std=BB_STD_DEV)
    upper_col, lower_col = robust_bbcols(bb, BB_PERIOD, BB_STD_DEV)
    df['bb_upper'] = bb[upper_col]
    df['bb_lower'] = bb[lower_col]

    high14 = df['high'].rolling(14).max()
    low14 = df['low'].rolling(14).min()
    df['wpr'] = (high14 - df['close']) / (high14 - low14) * -100

    atr = ta.atr(df['high'], df['low'], df['close'], length=CE_ATR_PERIOD)
    high20 = df['high'].rolling(CE_ATR_PERIOD, min_periods=1).max()
    df['chandelier_exit'] = high20 - CE_ATR_MULTIPLIER * atr

    st = ta.supertrend(df['high'], df['low'], df['close'], length=ST_LENGTH, multiplier=ST_MULTIPLIER)
    df['supertrend'] = st[f'SUPERT_{ST_LENGTH}_{ST_MULTIPLIER}']
    df['supertrend_dir'] = st[f'SUPERTd_{ST_LENGTH}_{ST_MULTIPLIER}']

    df = df.dropna(subset=['donchian_high','ema200','adx','vol_sma20','bb_upper','bb_lower','wpr',
                           'chandelier_exit','supertrend']).reset_index(drop=True)
    return df

def breakout_signal(df):
    cond_daily = (df['close'] > df['donchian_high'].shift(1))
    cond_vol = (df['volume'] > VOLUME_MULT * df['vol_sma20'])
    cond_weekly = True
    if 'weekly_donchian_high' in df.columns:
        cond_weekly = (df['close'] > df['weekly_donchian_high'].shift(1))
    df['breakout_signal'] = (cond_daily & cond_vol & cond_weekly).astype(int)
    return df

def bollinger_breakout_signal(df):
    df['bb_breakout_signal'] = (
        (df['close'] > df['bb_upper']) &
        (df['volume'] > VOLUME_MULT * df['vol_sma20'])
    ).astype(int)
    return df

def bollinger_pullback_signal(df):
    cond_pull = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    return df

def wpr_signal(df):
    cond_buy = (df['wpr'] < -80) & (df['wpr'].shift(1) >= -80)
    df['wpr_buy_signal'] = cond_buy.astype(int)
    return df

def combine_entry_signals(df):
    # Chandelier entry confirm: only signals where current close > chandelier_exit
    chandelier_confirm = df['close'] > df['chandelier_exit']
    df['entry_signal'] = 0
    df['entry_type'] = ''
    df.loc[(df['breakout_signal'] == 1) & chandelier_confirm, ['entry_signal', 'entry_type']] = [1, 'Breakout']
    df.loc[(df['bb_breakout_signal'] == 1) & chandelier_confirm & (df['entry_signal'] == 0), ['entry_signal', 'entry_type']] = [1, 'BB_Breakout']
    df.loc[(df['bb_pullback_signal'] == 1) & chandelier_confirm & (df['entry_signal'] == 0), ['entry_signal', 'entry_type']] = [1, 'BB_Pullback']
    df.loc[(df['wpr_buy_signal'] == 1) & chandelier_confirm & (df['entry_signal'] == 0), ['entry_signal', 'entry_type']] = [1, 'W%R_Buy']
    return df

def add_ml_features(df):
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['volumechange'] = df['volume'].pct_change().fillna(0)
    df = df.dropna(subset=['rsi','atr','adx','ema10','ema21','volumechange']).reset_index(drop=True)
    return df

def ml_filter(df, model):
    features = ['rsi','atr','adx','ema10','ema21','volumechange']
    df = add_ml_features(df)
    if df.empty:
        return df
    X = df[features]
    if X.shape[0] == 0:
        return df
    try:
        df['ml_signal'] = model.predict(X)
    except Exception as e:
        print(f"[ML ERROR] {e}")
        return df
    df['entry_signal'] = np.where((df['entry_signal']==1)&(df['ml_signal']==1), 1, 0)
    df = df[df['entry_signal']==1].reset_index(drop=True)
    return df

def regime_filter(df):
    cond = (df['close'] > df['ema200']) & \
           (df['adx'] > ADX_THRESHOLD) & \
           (df['volume'] > VOLUME_MULT * df['vol_sma20'])
    return df[cond].reset_index(drop=True)

def backtest(df, symbol):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        date = row['date']
        price = row['close']
        sig = row['entry_signal']
        sigtype = row['entry_type']

        to_close = []
        for pid, pos in positions.items():
            ret = (price - pos['entry_price']) / pos['entry_price']
            if price > pos['high']:
                pos['high'] = price
            chandelier_stop = df.loc[i, 'chandelier_exit']
            supertrend_stop = df.loc[i, 'supertrend']
            trail_stop = max(chandelier_stop, supertrend_stop)
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = trail_stop
            if pos['trail_active'] and trail_stop > pos['trail_stop']:
                pos['trail_stop'] = trail_stop
            exit_condition = ret >= PROFIT_TARGET or ret <= -STOP_LOSS or sig == 0 or \
                             (pos['trail_active'] and price <= pos['trail_stop']) or \
                             (row['supertrend_dir'] == -1)
            if exit_condition:
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                charges = calc_charges(buy_val, sell_val)
                exit_val = sell_val * (1 - TRANSACTION_COST)
                pnl = exit_val - buy_val - charges
                exit_reason = (
                    'Profit Target' if ret >= PROFIT_TARGET else
                    'Stop Loss' if ret <= -STOP_LOSS else
                    'Signal Exit' if sig == 0 else
                    'Chandelier Exit' if price <= pos['trail_stop'] else
                    'SuperTrend Exit' if row['supertrend_dir'] == -1 else
                    'Other'
                )
                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'days_held': (date - pos['entry_date']).days,
                    'pnl': pnl,
                    'return_pct': ret * 100,
                    'entry_type': pos['entry_type'],
                    'exit_reason': exit_reason
                })
                cash += exit_val
                to_close.append(pid)
                trade_count += 1
                if trade_count >= MAX_TRADES:
                    break
        for pid in to_close:
            positions.pop(pid)

        if trade_count >= MAX_TRADES:
            break

        if sig == 1 and len(positions) < MAX_POSITIONS and cash >= POSITION_SIZE:
            shares = POSITION_SIZE / price
            positions[len(positions)+1] = {
                'entry_date': date,
                'entry_price': price,
                'shares': shares,
                'high': price,
                'trail_active': False,
                'trail_stop': 0,
                'entry_type': sigtype
            }
            cash -= POSITION_SIZE * (1 + TRANSACTION_COST)

    return trades

if __name__ == "__main__":
    model = load_model()
    cutoff_date = datetime.now() - timedelta(days=5 * 365)
    all_trades = []

    for file in os.listdir(DATA_PATH):
        if not file.endswith(".csv"):
            continue
        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(DATA_PATH, file), parse_dates=['date'])
        df = df[df['date'] >= cutoff_date].reset_index(drop=True)
        if df.empty:
            continue

        df = add_indicators(df)
        df = breakout_signal(df)
        df = bollinger_breakout_signal(df)
        df = bollinger_pullback_signal(df)
        df = wpr_signal(df)
        df = combine_entry_signals(df)
        df = regime_filter(df)

        if USE_ML_CONFIRM and model is not None:
            df = ml_filter(df, model)

        if df.empty:
            continue

        trades = backtest(df, symbol)
        all_trades.extend(trades)

    cols = ['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'days_held', 'pnl', 'return_pct', 'entry_type', 'exit_reason']
    log_df = pd.DataFrame(all_trades, columns=cols)
    log_df.to_csv(TRADE_LOG_CSV, index=False)

    print(f"Total Trades: {len(log_df)}")
    print("P&L by entry_type:")
    print(log_df.groupby('entry_type')['pnl'].agg(['count', 'sum', 'mean']))
    print("Overall PnL:", log_df['pnl'].sum())
    print("Win Rate: {:.2f}%".format((log_df['pnl'] > 0).mean() * 100))

    # ==== NEW EXIT REASON STATS ====
    print("\nExit reason performance:")
    print(log_df.groupby('exit_reason')['pnl'].agg(['count','sum','mean']).sort_values('sum', ascending=False))
