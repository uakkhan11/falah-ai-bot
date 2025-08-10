#!/usr/bin/env python3
import os, pandas as pd, numpy as np
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta
import itertools
from typing import Dict, List, Tuple

# ==== CONFIGURATION ====
BASE_DIR = "/root/falah-ai-bot"
DATA_DIRS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

# Parameters for optimization
PARAM_GRID = {
    'ATR_MULTIPLIER': [1.5, 2.0, 2.5, 3.0],
    'ATR_PERIOD': [14, 20, 22],
    'PROFIT_TARGET': [0.08, 0.10, 0.12, 0.15],
    'VOLUME_MULT': [1.1, 1.2, 1.5],
    'ADX_THRESHOLD': [15, 20]
}

TIMEFRAME = 'daily'
DATA_PATH = DATA_DIRS[TIMEFRAME]
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Base strategy parameters
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE = 100_000
TRAIL_TRIGGER = 0.05
TRAIL_DISTANCE = 0.02
TRANSACTION_COST = 0.001
MAX_POSITIONS = 5
MAX_TRADES = 2000

# Cost structure
STT_RATE = 0.001
STAMP_DUTY_RATE = 0.00015
EXCHANGE_RATE = 0.0000345
GST_RATE = 0.18
SEBI_RATE = 0.000001
DP_CHARGE = 13.5

def calc_charges(buy_val, sell_val):
    stt = (buy_val + sell_val) * STT_RATE
    stamp = buy_val * STAMP_DUTY_RATE
    exch = (buy_val + sell_val) * EXCHANGE_RATE
    gst = exch * GST_RATE
    sebi = (buy_val + sell_val) * SEBI_RATE
    return stt + stamp + exch + gst + sebi + DP_CHARGE

def robust_bbcols(bb, period, std):
    upper = [c for c in bb.columns if c.startswith(f"BBU_{period}_")]
    lower = [c for c in bb.columns if c.startswith(f"BBL_{period}_")]
    if not upper: upper = [c for c in bb.columns if 'BBU' in c]
    if not lower: lower = [c for c in bb.columns if 'BBL' in c]
    return upper, lower

def add_indicators(df, atr_period=14):
    # --- Weekly Donchian for daily data ---
    if TIMEFRAME == 'daily':
        df_weekly = df.resample('W-MON', on='date').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        df_weekly['weekly_donchian_high'] = df_weekly['high'].rolling(20, 1).max()
        df['weekly_donchian_high'] = (
            df_weekly.set_index('date')['weekly_donchian_high']
            .reindex(df['date'], method='ffill').values
        )
    
    # --- Core Indicators ---
    df['donchian_high'] = df['high'].rolling(20, 1).max()
    df['ema200'] = ta.ema(df['close'], length=200)

    # --- ADX with None-check ---
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and 'ADX_14' in adx_df.columns:
            df['adx'] = adx_df['ADX_14']
        else:
            df['adx'] = np.nan
    except Exception:
        df['adx'] = np.nan

    # --- Volume SMA (safe) ---
    if 'volume' in df.columns:
        df['vol_sma20'] = df['volume'].rolling(20, min_periods=1).mean()
    else:
        df['vol_sma20'] = np.nan

    # --- Bollinger Bands (with robust column lookup) ---
    bb = ta.bbands(df['close'], length=20, std=2)
    ucol, lcol = robust_bbcols(bb, 20, 2)
    df['bb_upper'] = bb[ucol[0]] if ucol else np.nan
    df['bb_lower'] = bb[lcol[0]] if lcol else np.nan

    # --- Williams %R ---
    high14 = df['high'].rolling(14).max()
    low14 = df['low'].rolling(14).min()
    df['wpr'] = (high14 - df['close']) / (high14 - low14) * -100
    
    # --- ATR for volatility stops ---
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)

    # --- Chandelier Exit (ATR-based trailing stop) ---
    atr_ce = ta.atr(df['high'], df['low'], df['close'], length=22)
    high20 = df['high'].rolling(22, 1).max()
    df['chandelier_exit'] = high20 - 3.0 * atr_ce

    return df.reset_index(drop=True)


def generate_signals(df, volume_mult=1.2):
    # Breakout signals
    cond_daily = df['close'] > df['donchian_high'].shift(1)
    cond_vol = df['volume'] > volume_mult * df['vol_sma20']
    cond_weekly = df['close'] > df['weekly_donchian_high'].shift(1) if 'weekly_donchian_high' in df else True
    df['breakout_signal'] = (cond_daily & cond_vol & cond_weekly).astype(int)
    
    # BB signals
    df['bb_breakout_signal'] = ((df['close'] > df['bb_upper']) & (df['volume'] > volume_mult * df['vol_sma20'])).astype(int)
    cond_pull = df['close'] < df['bb_lower']
    cond_resume = df['close'] > df['bb_lower'].shift(1)
    df['bb_pullback_signal'] = (cond_pull.shift(1) & cond_resume).astype(int)
    
    # W%R signal
    df['wpr_buy_signal'] = ((df['wpr'] < -80) & (df['wpr'].shift(1) >= -80)).astype(int)
    
    return df

def combine_signals(df, adx_threshold=15, volume_mult=1.2):
    # Regime filter
    regime_mask = (df['close'] > df['ema200']) & (df['adx'] > adx_threshold) & (df['volume'] > volume_mult * df['vol_sma20'])
    
    # Chandelier entry confirmation
    chandelier_confirm = df['close'] > df['chandelier_exit']
    
    df['entry_signal'] = 0
    df['entry_type'] = ''
    
    # Priority-based signal combination with regime and Chandelier filters
    mask = regime_mask & chandelier_confirm
    
    df.loc[(df['breakout_signal'] == 1) & mask, ['entry_signal', 'entry_type']] = [1, 'Breakout']
    df.loc[(df['bb_breakout_signal'] == 1) & mask & (df['entry_signal'] == 0), ['entry_signal', 'entry_type']] = [1, 'BB_Breakout']
    df.loc[(df['bb_pullback_signal'] == 1) & mask & (df['entry_signal'] == 0), ['entry_signal', 'entry_type']] = [1, 'BB_Pullback']
    df.loc[(df['wpr_buy_signal'] == 1) & mask & (df['entry_signal'] == 0), ['entry_signal', 'entry_type']] = [1, 'W%R_Buy']
    
    return df

def apply_ml_filter(df, model, use_ai=True):
    if not use_ai or model is None:
        df['ml_signal'] = 1
        df['ai_confirmed'] = 'N/A'
        return df
    
    # ML features
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr_ml'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['volumechange'] = df['volume'].pct_change().fillna(0)
    
    features = ['rsi', 'atr_ml', 'adx', 'ema10', 'ema21', 'volumechange']
    df = df.dropna(subset=features).reset_index(drop=True)
    
    if df.empty:
        return df
    
    try:
        X = df[features]
        df['ml_signal'] = model.predict(X)
        df['ai_confirmed'] = np.where((df['entry_signal'] == 1) & (df['ml_signal'] == 1), 'Yes', 'No')
        df['entry_signal'] = np.where((df['entry_signal'] == 1) & (df['ml_signal'] == 1), 1, 0)
    except Exception as e:
        print(f"ML Error: {e}")
        df['ml_signal'] = 1
        df['ai_confirmed'] = 'Error'
    
    return df

def categorize_trade_duration(days):
    if days <= 1:
        return 'Day_Trade'
    elif days <= 7:
        return 'Short_Swing'
    elif days <= 30:
        return 'Medium_Swing'
    elif days <= 90:
        return 'Long_Swing'
    else:
        return 'Position_Trade'

def backtest_with_params(df, symbol, params, model=None, use_ai=True):
    # Extract parameters
    atr_mult = params['ATR_MULTIPLIER']
    atr_period = params['ATR_PERIOD']
    profit_target = params['PROFIT_TARGET']
    volume_mult = params['VOLUME_MULT']
    adx_threshold = params['ADX_THRESHOLD']
    
    # Add indicators and signals
    df = add_indicators(df, atr_period)
    df = generate_signals(df, volume_mult)
    df = combine_signals(df, adx_threshold, volume_mult)
    df = apply_ml_filter(df, model, use_ai)
    
    if df.empty:
        return []
    
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
        ai_conf = row.get('ai_confirmed', 'N/A')
        
        # Exit logic
        to_close = []
        for pid, pos in positions.items():
            ret = (price - pos['entry_price']) / pos['entry_price']
            
            # ATR-based stop loss
            atr_stop_price = pos['entry_price'] - (pos['entry_atr'] * atr_mult)
            
            if price > pos['high']:
                pos['high'] = price
            
            # Dynamic trailing stop
            chandelier_stop = row['chandelier_exit']
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = chandelier_stop
            
            if pos['trail_active'] and chandelier_stop > pos['trail_stop']:
                pos['trail_stop'] = chandelier_stop
            
            # Exit conditions
            exit_reason = None
            if ret >= profit_target:
                exit_reason = 'Profit Target'
            elif price <= atr_stop_price:
                exit_reason = 'ATR Stop Loss'
            elif sig == 0:
                exit_reason = 'Signal Exit'
            elif pos['trail_active'] and price <= pos['trail_stop']:
                exit_reason = 'Chandelier Exit'
            
            if exit_reason:
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                charges = calc_charges(buy_val, sell_val)
                pnl = sell_val * (1 - TRANSACTION_COST) - buy_val - charges
                days_held = (date - pos['entry_date']).days
                
                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'days_held': days_held,
                    'trade_category': categorize_trade_duration(days_held),
                    'pnl': pnl,
                    'return_pct': ret * 100,
                    'entry_type': pos['entry_type'],
                    'exit_reason': exit_reason,
                    'ai_confirmed': pos['ai_confirmed'],
                    'atr_stop_price': atr_stop_price,
                    'entry_atr': pos['entry_atr']
                })
                
                cash += sell_val * (1 - TRANSACTION_COST)
                to_close.append(pid)
                trade_count += 1
                
                if trade_count >= MAX_TRADES:
                    break
        
        for pid in to_close:
            positions.pop(pid)
        
        if trade_count >= MAX_TRADES:
            break
        
        # Entry logic
        if sig == 1 and len(positions) < MAX_POSITIONS and cash >= POSITION_SIZE:
            shares = POSITION_SIZE / price
            current_atr = row['atr']
            
            positions[len(positions) + 1] = {
                'entry_date': date,
                'entry_price': price,
                'entry_atr': current_atr,
                'shares': shares,
                'high': price,
                'trail_active': False,
                'trail_stop': 0,
                'entry_type': sigtype,
                'ai_confirmed': ai_conf
            }
            
            cash -= POSITION_SIZE * (1 + TRANSACTION_COST)
    
    return trades

def optimize_parameters(train_data, param_grid, model=None, use_ai=True):
    best_params = None
    best_score = -np.inf
    
    # Generate parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        
        all_trades = []
        for symbol_file in os.listdir(DATA_PATH):
            if not symbol_file.endswith('.csv'):
                continue
            
            symbol = symbol_file.replace('.csv', '')
            symbol_data = train_data[train_data['symbol'] == symbol].copy()
            
            if symbol_data.empty:
                continue
            
            trades = backtest_with_params(symbol_data, symbol, params, model, use_ai)
            all_trades.extend(trades)
        
        if all_trades:
            trade_df = pd.DataFrame(all_trades)
            # Score based on total PnL adjusted for drawdown
            total_pnl = trade_df['pnl'].sum()
            win_rate = (trade_df['pnl'] > 0).mean()
            score = total_pnl * win_rate  # Simple scoring function
            
            if score > best_score:
                best_score = score
                best_params = params
    
    return best_params if best_params else list(param_grid.values())[0]

def walk_forward_test(data, train_years=2, test_years=1):
    # Load model
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    
    results = []
    start_year = data['date'].dt.year.min()
    end_year = data['date'].dt.year.max()
    
    current_year = start_year + train_years
    
    while current_year + test_years <= end_year:
        # Training window
        train_data = data[data['date'].dt.year.between(current_year - train_years, current_year - 1)]
        
        # Test window
        test_data = data[data['date'].dt.year.between(current_year, current_year + test_years - 1)]
        
        if train_data.empty or test_data.empty:
            current_year += test_years
            continue
        
        print(f"Training: {current_year - train_years}-{current_year - 1}, Testing: {current_year}-{current_year + test_years - 1}")
        
        # Optimize parameters on training data
        best_params = optimize_parameters(train_data, PARAM_GRID, model, use_ai=True)
        
        # Test with AI
        all_trades_ai = []
        for symbol_file in os.listdir(DATA_PATH):
            if not symbol_file.endswith('.csv'):
                continue
            
            symbol = symbol_file.replace('.csv', '')
            symbol_data = test_data[test_data['symbol'] == symbol].copy()
            
            if symbol_data.empty:
                continue
            
            trades = backtest_with_params(symbol_data, symbol, best_params, model, use_ai=True)
            all_trades_ai.extend(trades)
        
        # Test without AI
        all_trades_no_ai = []
        for symbol_file in os.listdir(DATA_PATH):
            if not symbol_file.endswith('.csv'):
                continue
            
            symbol = symbol_file.replace('.csv', '')
            symbol_data = test_data[test_data['symbol'] == symbol].copy()
            
            if symbol_data.empty:
                continue
            
            trades = backtest_with_params(symbol_data, symbol, best_params, model, use_ai=False)
            all_trades_no_ai.extend(trades)
        
        # Store results
        results.append({
            'period': f"{current_year}-{current_year + test_years - 1}",
            'best_params': best_params,
            'trades_with_ai': all_trades_ai,
            'trades_without_ai': all_trades_no_ai
        })
        
        current_year += test_years
    
    return results

def analyze_results(results):
    print("=== WALK-FORWARD TEST RESULTS ===\n")
    
    for result in results:
        period = result['period']
        trades_ai = pd.DataFrame(result['trades_with_ai']) if result['trades_with_ai'] else pd.DataFrame()
        trades_no_ai = pd.DataFrame(result['trades_without_ai']) if result['trades_without_ai'] else pd.DataFrame()
        
        print(f"Period: {period}")
        print(f"Best Parameters: {result['best_params']}")
        
        if not trades_ai.empty:
            print("WITH AI:")
            print(f"  Total Trades: {len(trades_ai)}")
            print(f"  Total PnL: ₹{trades_ai['pnl'].sum():,.2f}")
            print(f"  Win Rate: {(trades_ai['pnl'] > 0).mean()*100:.2f}%")
            print("  By Trade Category:")
            print(trades_ai.groupby('trade_category')['pnl'].agg(['count', 'sum', 'mean']))
            print("  By Entry Type:")
            print(trades_ai.groupby('entry_type')['pnl'].agg(['count', 'sum', 'mean']))
        
        if not trades_no_ai.empty:
            print("\nWITHOUT AI:")
            print(f"  Total Trades: {len(trades_no_ai)}")
            print(f"  Total PnL: ₹{trades_no_ai['pnl'].sum():,.2f}")
            print(f"  Win Rate: {(trades_no_ai['pnl'] > 0).mean()*100:.2f}%")
        
        print("-" * 80)

if __name__ == "__main__":
    # Load all data
    all_data = []
    for file in os.listdir(DATA_PATH):
        if not file.endswith(".csv"):
            continue
        
        symbol = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(DATA_PATH, file), parse_dates=['date'])
        df['symbol'] = symbol
        all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    cutoff_date = datetime.now() - timedelta(days=5 * 365)
    combined_data = combined_data[combined_data['date'] >= cutoff_date]
    
    print("Starting Walk-Forward Testing...")
    results = walk_forward_test(combined_data)
    analyze_results(results)
