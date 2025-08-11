# strategy_utils.py

import pandas as pd
import numpy as np
import pandas_ta as ta

# Constants — match these to your config.py values
ATR_PERIOD = 14
ADX_THRESHOLD_DEFAULT = 15
ADX_THRESHOLD_BREAKOUT = 12
VOLUME_MULT_DEFAULT = 1.2
VOLUME_MULT_BREAKOUT = 1.1

def robust_bbcols(bb):
    """Get Bollinger Band upper and lower column names safely."""
    if bb is None or bb.empty:
        return None, None
    ucol = [c for c in bb.columns if 'BBU' in c][0]
    lcol = [c for c in bb.columns if 'BBL' in c][0]
    return ucol, lcol

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all technical indicator columns required by the strategy."""
    if 'date' in df.columns:
        df = df.sort_values('date')
    df = df.copy()

    # Weekly Donchian high (20-week high)
    try:
        df_weekly = df.resample('W-MON', on='date').agg({
            'open': 'first',
            'high': 'max',
            'low':  'min',
            'close':'last',
            'volume':'sum'
        }).dropna().reset_index()
        df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20, min_periods=1).max()
        df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'] \
            .reindex(df['date'], method='ffill').values
    except Exception:
        df['weekly_donchian_high'] = np.nan

    # Donchian channel 20-day high
    df['donchian_high'] = df['high'].rolling(20, min_periods=1).max()

    # EMA200
    df['ema200'] = ta.ema(df['close'], length=200)

    # ADX
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df else np.nan
    except Exception:
        df['adx'] = np.nan

    # Volume SMA20
    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()

    # Bollinger Bands
    try:
        bb = ta.bbands(df['close'], length=20, std=2)
        ucol, lcol = robust_bbcols(bb)
        if ucol and lcol:
            df['bb_upper'] = bb[ucol]
            df['bb_lower'] = bb[lcol]
        else:
            df['bb_upper'] = np.nan
            df['bb_lower'] = np.nan
    except Exception:
        df['bb_upper'] = np.nan
        df['bb_lower'] = np.nan

    # Williams %R (14)
    high14 = df['high'].rolling(14, min_periods=1).max()
    low14  = df['low'].rolling(14, min_periods=1).min()
    df['wpr'] = (high14 - df['close']) / (high14 - low14) * -100

    # ATR (14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    # Chandelier Exit
    atr_ce = ta.atr(df['high'], df['low'], df['close'], length=22)
    high20 = df['high'].rolling(22, min_periods=1).max()
    df['chandelier_exit'] = high20 - 3.0 * atr_ce

    # Supertrend
    try:
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
        df['supertrend'] = st['SUPERT_10_3.0']
        df['supertrend_dir'] = st['SUPERTd_10_3.0']
    except Exception:
        df['supertrend'] = np.nan
        df['supertrend_dir'] = 0

    # Ensure required cols exist
    cols = [
        'close', 'donchian_high', 'ema200', 'adx', 'vol_sma20',
        'bb_upper', 'bb_lower', 'wpr', 'atr', 'chandelier_exit',
        'supertrend', 'supertrend_dir', 'weekly_donchian_high'
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    return df.reset_index(drop=True)

# === Signal functions ===

def breakout_signal(df):
    cond_d = df['close'] > df['donchian_high'].shift(1)
    cond_v = df['volume'] > VOLUME_MULT_BREAKOUT * df['vol_sma20']
    cond_w = df['close'] > df['weekly_donchian_high'].shift(1)
    df['breakout_signal'] = (cond_d & cond_v & cond_w).astype(int)
    return df

def bb_breakout_signal(df):
    df['bb_breakout_signal'] = (
        (df['close'] > df['bb_upper']) &
        (df['volume'] > VOLUME_MULT_BREAKOUT * df['vol_sma20'])
    ).astype(int)
    return df

def bb_pullback_signal(df):
    cond_pull = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    return df

def combine_signals(df):
    chand_or_st = (df['close'] > df['chandelier_exit']) | (df['supertrend_dir'] == 1)
    
    regime_breakout = df['close'].gt(df['ema200'], fill_value=False) & df['adx'].gt(ADX_THRESHOLD_BREAKOUT, fill_value=False)
    regime_default = df['close'].gt(df['ema200'], fill_value=False) & df['adx'].gt(ADX_THRESHOLD_DEFAULT, fill_value=False)

    df['entry_signal'] = 0
    df['entry_type'] = ''
    df.loc[(df['breakout_signal'] == 1) & chand_or_st & regime_breakout,
           ['entry_signal', 'entry_type']] = [1, 'Breakout']
    df.loc[(df['bb_breakout_signal'] == 1) & chand_or_st & regime_breakout & (df['entry_signal'] == 0),
           ['entry_signal', 'entry_type']] = [1, 'BB_Breakout']
    df.loc[(df['bb_pullback_signal'] == 1) & chand_or_st & regime_default & (df['entry_signal'] == 0),
           ['entry_signal', 'entry_type']] = [1, 'BB_Pullback']
    return df
