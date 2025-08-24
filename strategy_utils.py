import pandas as pd
import numpy as np
import pandas_ta as ta

# Constants
ATR_PERIOD = 14
ADX_THRESHOLD_DEFAULT = 15
ADX_THRESHOLD_BREAKOUT = 12
VOLUME_MULT_DEFAULT = 1.2
VOLUME_MULT_BREAKOUT_BASE = 1.1

def robust_bbcols(bb):
    if bb is None or bb.empty:
        return None, None
    ucol = [c for c in bb.columns if 'BBU' in c][0]
    lcol = [c for c in bb.columns if 'BBL' in c][0]
    return ucol, lcol

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' in df.columns:
        df = df.sort_values('date')
    df = df.copy()
    try:
        df_weekly = df.resample('W-MON', on='date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20, min_periods=1).max()
        df['weekly_donchian_high'] = df_weekly.set_index('date')['weekly_donchian_high'] \
            .reindex(df['date'], method='ffill').values
    except Exception:
        df['weekly_donchian_high'] = np.nan

    df['donchian_high'] = df['high'].rolling(20, min_periods=1).max()
    df['ema200'] = ta.ema(df['close'], length=200)
    df['ema200_slope'] = df['ema200'].diff()

    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=ATR_PERIOD)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df else np.nan
    except Exception:
        df['adx'] = np.nan

    df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()

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

    return df.reset_index(drop=True)

def breakout_signal(df):
    cond_d = df['close'] > df['donchian_high'].shift(1)
    cond_v = df['volume'] > VOLUME_MULT_BREAKOUT_BASE * df['vol_sma20']
    cond_w = df['close'] > df['weekly_donchian_high'].shift(1)
    df['breakout_signal'] = (cond_d & cond_v & cond_w).astype(int)
    return df

def adaptive_volume_multiplier(df, window=10):
    # Adaptive volume multiplier based on short-term average volume changes
    vol_change = df['volume'].pct_change().rolling(window).mean()
    # Higher multiplier when volume is rising
    return np.where(vol_change > 0, VOLUME_MULT_BREAKOUT_BASE + 0.1, VOLUME_MULT_BREAKOUT_BASE)

def refined_bb_breakout_signal(df):
    vol_mult = adaptive_volume_multiplier(df)
    cond_close = df['close'] > df['bb_upper']
    cond_vol = df['volume'] > vol_mult * df['vol_sma20']
    cond_adx = df['adx'] > ADX_THRESHOLD_BREAKOUT
    df['bb_breakout_signal'] = (cond_close & cond_vol & cond_adx).astype(int)
    return df

def refined_bb_pullback_signal(df):
    cond_pull = df['close'].shift(1) < df['bb_lower'].shift(1)
    cond_resume = df['close'] > df['bb_lower']
    cond_adx = df['adx'] > ADX_THRESHOLD_DEFAULT
    df['bb_pullback_signal'] = (cond_pull & cond_resume & cond_adx).astype(int)
    return df

def combine_signals(df):
    chand_or_st = (df['close'] > df.get('chandelier_exit', float('-inf'))) | (df.get('supertrend_dir', 0) == 1)

    regime_breakout = df['close'].gt(df['ema200'], fill_value=False) & df['adx'].gt(ADX_THRESHOLD_BREAKOUT, fill_value=False) & (df['ema200_slope'] > 0)
    regime_default = df['close'].gt(df['ema200'], fill_value=False) & df['adx'].gt(ADX_THRESHOLD_DEFAULT, fill_value=False) & (df['ema200_slope'] > 0)

    df = breakout_signal(df)
    df = refined_bb_breakout_signal(df)
    df = refined_bb_pullback_signal(df)

    df['entry_signal'] = 0
    df['entry_type'] = ''

    df.loc[(df['breakout_signal'] == 1) & chand_or_st & regime_breakout,
           ['entry_signal', 'entry_type']] = [1, 'Breakout']

    df.loc[(df['bb_breakout_signal'] == 1) & chand_or_st & regime_breakout & (df['entry_signal'] == 0),
           ['entry_signal', 'entry_type']] = [1, 'BB_Breakout']

    df.loc[(df['bb_pullback_signal'] == 1) & chand_or_st & regime_default & (df['entry_signal'] == 0),
           ['entry_signal', 'entry_type']] = [1, 'BB_Pullback']

    return df
