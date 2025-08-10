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
USE_ML_CONFIRM = True  # Toggle AI confirmation

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

# Chandelier Exit Parameters
CE_ATR_PERIOD      = 22
CE_ATR_MULTIPLIER  = 3.0

# SuperTrend Parameters
ST_LENGTH          = 10
ST_MULTIPLIER      = 3.0

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

# ===== INDICATORS & SIGNAL COMPUTATION =====
def add_indicators(df):
    # Weekly Donchian High (for daily timeframe only)
    if TIMEFRAME == 'daily':
        df_weekly = df.resample('W-MON', on='date').agg({
            'open':'first','high':'max','low':'min','close':'last','volume':'sum'
        }).dropna().reset_index()
        df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20, min_periods=1).max()
        df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'].reindex(df['date'], method='ffill').values

    # Donchian High 20-day rolling max
    if 'high' in df.columns and len(df) >= 20:
        df['donchian_high'] = df['high'].rolling(20, min_periods=1).max()
    else:
        df['donchian_high'] = np.nan

    df['ema200'] = ta.ema(df['close'], length=200)
    df['adx'] = np.nan
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and 'ADX_14' in adx_df.columns:
            df['adx'] = adx_df['ADX_14']
    except Exception:
        pass
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()

    # Bollinger bands
    bb = ta.bbands(df['close'], length=BB_PERIOD, std=BB_STD_DEV)
    upper_col, lower_col = robust_bbcols(bb, BB_PERIOD, BB_STD_DEV)
    df['bb_upper'] = bb[upper_col]
    df['bb_lower'] = bb[lower_col]

    # W%R
    high14 = df['high'].rolling(14).max()
    low14 = df['low'].rolling(14).min()
    df['wpr'] = (high14 - df['close']) / (high14 - low14) * -100

    # Chandelier Exit (long stop)
    atr = ta.atr(df['high'], df['low'], df['close'], length=CE_ATR_PERIOD)
    high20 = df['high'].rolling(CE_ATR_PERIOD, min_periods=1).max()
    df['chandelier_exit'] = high20 - CE_ATR_MULTIPLIER * atr

    # SuperTrend (using pandas_ta)
    try:
        st = ta.supertrend(df['high'], df['low'], df['close'], length=ST_LENGTH, multiplier=ST_MULTIPLIER)
        df['supertrend'] = st[f'SUPERT_{ST_LENGTH}_{ST_MULTIPLIER}']
        df['supertrend_dir'] = st[f'SUPERTd_{ST_LENGTH}_{ST_MULTIPLIER}']  # Direction: 1 bullish, -1 bearish
    except Exception:
        df['supertrend'] = np.nan
        df['supertrend_dir'] = 0

    # drop rows with essential NaNs
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
    df['bb_pullback_signal'] = df['bb_pullback_signal'].fillna(0)
    return df

def wpr_signal(df):
    cond_buy = (df['wpr'] < -80) & (df['wpr'].shift(1) >= -80)
    cond_sell = (df['wpr'] > -20) & (df['wpr'].shift(1) <= -20)
    df['wpr_buy_signal'] = cond_buy.astype(int)
    df['wpr_sell_signal'] = cond_sell.astype(int)
    return df

def combine_entry_signals(df):
    df['entry_signal'] = 0
    df['entry_type'] = ''
    # Priority order (can be tuned)
    df.loc[df['breakout_signal'] == 1, ['entry_signal', 'entry_type']] = [1, 'Breakout']
    df.loc[(df['entry_signal'] == 0) & (df['bb_breakout_signal'] == 1), ['entry_signal', 'entry_type']] = [1, 'BB_Breakout']
    df.loc[(df['entry_signal'] == 0) & (df['bb_pullback_signal'] == 1), ['entry_signal', 'entry_type']] = [1, 'BB_Pullback']
    df.loc[(df['entry_signal'] == 0) & (df['wpr_buy_signal'] == 1), ['entry_signal', 'entry_type']] = [1, 'W%R_Buy']
    return df

# ===== AI FEATURES AND FILTER =====
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
        df['ml_signal'] = pd.Series([0]*len(df),dtype=int)
        return df
    X = df[features]
    if X.shape[0] == 0:
        df['ml_signal'] = pd.Series([0]*len(df),dtype=int)
        return df
    try:
        df['ml_signal'] = model.predict(X)
    except Exception as e:
        print(f"[ML ERROR] {e}")
        df['ml_signal'] = pd.Series([0]*len(df),dtype=int)
    # Filter entry_signal with ml_signal
    df['entry_signal'] = np.where((df['entry_signal']==1)&(df['ml_signal']==1), 1, 0)
    df = df[df['entry_signal']==1].reset_index(drop=True)
    return df

# ===== REGIME FILTER =====
def regime_filter(df):
    cond = (df['close'] > df['ema200']) & \
           (df['adx'] > ADX_THRESHOLD) & \
           (df['volume'] > VOLUME_MULT * df['vol_sma20'])
    return df[cond].reset_index(drop=True)

# ===== BACKTEST ENGINE =====
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

        # EXIT LOGIC
        to_close = []
        for pid, pos in positions.items():
            ret = (price - pos['entry_price']) / pos['entry_price']

            # Update highest price for trailing stops
            if price > pos['high']:
                pos['high'] = price

            # Compute trailing stops using Chandelier Exit and SuperTrend
            chandelier_stop = df.loc[i, 'chandelier_exit']
            supertrend_stop = df.loc[i, 'supertrend']

            # Conditions for trailing stops use lower of two stop levels as exit threshold
            trail_stop = max(chandelier_stop, supertrend_stop) if not np.isnan(chandelier_stop) and not np.isnan(supertrend_stop) else \
                         chandelier_stop if not np.isnan(chandelier_stop) else \
                         supertrend_stop if not np.isnan(supertrend_stop) else 0

            # Activate trailing if profit > trigger
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = trail_stop

            # Adjust trailing stop if it moves up
            if pos['trail_active'] and trail_stop > pos['trail_stop']:
                pos['trail_stop'] = trail_stop

            # Exit Conditions:
            # 1) Profit Target
            # 2) Stop Loss
            # 3) Entry Signal turned off (sig==0)
            # 4) Price hits trailing stop
            # 5) Price closes below SuperTrend line (bearish flip)
            exit_condition = ret >= PROFIT_TARGET or ret <= -STOP_LOSS or sig == 0 or \
                             (pos['trail_active'] and price <= pos['trail_stop']) or \
                             (row['supertrend_dir'] == -1)  # SuperTrend flip bearish exit

            if exit_condition:
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                charges = calc_charges(buy_val, sell_val)
                exit_val = sell_val * (1 - TRANSACTION_COST)
                pnl = exit_val - buy_val - charges
                exit_reason = ''
                if ret >= PROFIT_TARGET:
                    exit_reason = 'Profit Target'
                elif ret <= -STOP_LOSS:
                    exit_reason = 'Stop Loss'
                elif sig == 0:
                    exit_reason = 'Signal Exit'
                elif pos['trail_active'] and price <= pos['trail_stop']:
                    exit_reason = 'Trailing Stop'
                elif row['supertrend_dir'] == -1:
                    exit_reason = 'SuperTrend Exit'
                else:
                    exit_reason = 'Unknown Exit'

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

        # ENTRY LOGIC
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
            cost = POSITION_SIZE * (1 + TRANSACTION_COST)
            cash -= cost

    # Final forced exit on last day for open positions
    last_date = df.iloc[-1]['date']
    last_price = df.iloc[-1]['close']
    for pos in positions.values():
        days_held = (last_date - pos['entry_date']).days
        if days_held >= 1:
            buy_val = pos['shares'] * pos['entry_price']
            sell_val = pos['shares'] * last_price
            charges = calc_charges(buy_val, sell_val)
            exit_val = sell_val * (1 - TRANSACTION_COST)
            pnl = exit_val - buy_val - charges
            ret = (last_price - pos['entry_price']) / pos['entry_price']
            trades.append({
                'symbol': symbol,
                'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                'exit_date': last_date.strftime('%Y-%m-%d'),
                'entry_price': pos['entry_price'],
                'exit_price': last_price,
                'days_held': days_held,
                'pnl': pnl,
                'return_pct': ret * 100,
                'entry_type': pos['entry_type'],
                'exit_reason': 'EOD Exit'
            })
            cash += exit_val

    return trades

# ===== MAIN EXECUTION =====
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

    # Save and Summarize
    cols = ['symbol', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'days_held', 'pnl', 'return_pct', 'entry_type', 'exit_reason']
    log_df = pd.DataFrame(all_trades, columns=cols)
    log_df.to_csv(TRADE_LOG_CSV, index=False)

    print(f"Total Trades: {len(log_df)}")
    print("P&L by entry_type:")
    print(log_df.groupby('entry_type')['pnl'].agg(['count', 'sum', 'mean']))
    print("Overall PnL:", log_df['pnl'].sum())
    print("Win Rate: {:.2f}%".format((log_df['pnl'] > 0).mean() * 100))
