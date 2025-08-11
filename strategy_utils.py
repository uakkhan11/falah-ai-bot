# strategy_utils.py

import pandas as pd
import numpy as np
import pandas_ta as ta

def add_indicators(df):
    # Your exact backtest code here, for example:
    if 'date' in df.columns:
        df = df.sort_values('date')
    # … continue with weekly donchian, EMA200, ADX, BB, WPR, ATR, Supertrend, etc.
    # Finally:
    return df.reset_index(drop=True)

def breakout_signal(df):
    cond_d = df['close'] > df['donchian_high'].shift(1)
    cond_v = df['volume'] > VOLUME_MULT_BREAKOUT * df['vol_sma20']
    cond_w = df['close'] > df['weekly_donchian_high'].shift(1)
    df['breakout_signal'] = (cond_d & cond_v & cond_w).astype(int)
    return df

def bb_breakout_signal(df):
    df['bb_breakout_signal'] = ((df['close'] > df['bb_upper']) &
                                (df['volume'] > VOLUME_MULT_BREAKOUT * df['vol_sma20'])).astype(int)
    return df

def bb_pullback_signal(df):
    cond_pull = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    return df

def combine_signals(df):
    chand_or_st = (df['close'] > df['chandelier_exit']) | (df['supertrend_dir'] == 1)
    regime_breakout = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD_BREAKOUT)
    regime_default = (df['close'] > df['ema200']) & (df['adx'] > ADX_THRESHOLD_DEFAULT)

    df['entry_signal'] = 0; df['entry_type'] = ''
    df.loc[(df['breakout_signal']==1) & chand_or_st & regime_breakout,
           ['entry_signal','entry_type']] = [1,'Breakout']
    df.loc[(df['bb_breakout_signal']==1) & chand_or_st & regime_breakout & (df['entry_signal']==0),
           ['entry_signal','entry_type']] = [1,'BB_Breakout']
    df.loc[(df['bb_pullback_signal']==1) & chand_or_st & regime_default & (df['entry_signal']==0),
           ['entry_signal','entry_type']] = [1,'BB_Pullback']
    return df
