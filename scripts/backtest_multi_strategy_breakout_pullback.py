#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta

# ---- CONFIG ----
BASE_DIR = "/root/falah-ai-bot"
DAILY_DIR = os.path.join(BASE_DIR, "swing_data")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TRADE_LOG_CSV = os.path.join(BASE_DIR, "backtest_trade_log_multi.csv")

# Edit this to your best NIFTY symbols
SYMBOLS_TO_TEST = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
USE_ML_CONFIRM = True

# Strategy params
INITIAL_CAPITAL = 1_000_000
POSITION_SIZE = 100_000
PROFIT_TARGET = 0.12
STOP_LOSS = 0.05
TRAIL_TRIGGER = 0.06
TRAIL_DISTANCE = 0.025
TRANSACTION_COST = 0.001
MAX_POSITIONS = 5
MAX_TRADES = 200

# CNC cost breakdown
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
    dp = DP_CHARGE
    return stt + stamp + exch + gst + sebi + dp

def add_indicators(df):
    df['ema200'] = ta.ema(df['close'], length=200)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else np.nan
    df['vol_sma20'] = df['volume'].rolling(20).mean()
    df['donchian_high'] = df['high'].rolling(20).max()
    return df

def willr_signals(df):
    high14 = df['high'].rolling(14).max()
    low14 = df['low'].rolling(14).min()
    df['w%r'] = (high14 - df['close']) / (high14 - low14) * -100
    buy = (df['w%r'] < -80) & (df['w%r'].shift(1) >= -80)
    df['wpr_signal'] = 0
    df.loc[buy, 'wpr_signal'] = 1
    return df

def breakout_signals(df):
    breakout = (df['close'] > df['donchian_high'].shift(1)) & (df['volume'] > 1.5 * df['vol_sma20'])
    df['breakout_signal'] = breakout.astype(int)
    return df

def pullback_signals(df):
    # price pulls back from 20EMA, stays above 50EMA, then resumes up with volume
    pullback = (df['close'] < df['ema20']) & (df['close'] > df['ema50']) & (df['volume'] > 1.1 * df['vol_sma20'])
    # Confirm resumption: use Williams %R buy signal or a jump above 20EMA with volume
    resume = (df['close'] > df['ema20']) & (df['volume'] > 1.2 * df['vol_sma20'])
    df['pullback_signal'] = (pullback.shift(1) & resume).astype(int)
    return df

def add_ml_features(df):
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['ema10'] = ta.ema(df['close'], length=10)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['volumechange'] = df['volume'].pct_change().fillna(0)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14'] if adx_df is not None and 'ADX_14' in adx_df.columns else np.nan
    return df

def ml_signals(df, model):
    features = ['rsi','atr','adx','ema10','ema21','volumechange']
    df = add_ml_features(df).dropna(subset=features).reset_index(drop=True)
    X = df[features]
    df['ml_signal'] = model.predict(X)
    return df

def regime_filter(df):
    cond = (df['close'] > df['ema200']) & (df['adx'] > 20) & (df['volume'] > 1.5 * df['vol_sma20'])
    return df[cond].reset_index(drop=True)

def combine_signals(df):
    # Combine three: W%R, Breakout, Pullback, keep 'signal_type'
    df = df.copy()
    df['entry_signal'] = 0
    df['signal_type'] = ""
    # Priority: breakout > pullback > W%R
    df.loc[df['breakout_signal'] == 1, 'entry_signal'] = 1
    df.loc[df['breakout_signal'] == 1, 'signal_type'] = "Breakout"
    # Pullback
    df.loc[(df['breakout_signal'] != 1) & (df['pullback_signal'] == 1), 'entry_signal'] = 1
    df.loc[(df['breakout_signal'] != 1) & (df['pullback_signal'] == 1), 'signal_type'] = "Pullback"
    # W%R oversold
    df.loc[(df['breakout_signal'] != 1) & (df['pullback_signal'] != 1) & (df['wpr_signal'] == 1), 'entry_signal'] = 1
    df.loc[(df['breakout_signal'] != 1) & (df['pullback_signal'] != 1) & (df['wpr_signal'] == 1),'signal_type'] = "W%R"
    return df

def backtest(df, symbol):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0

    for i in range(1, len(df)):
        date = df.at[i, 'date']
        price = df.at[i, 'close']
        sig = df.at[i, 'entry_signal']
        sigtype = df.at[i, 'signal_type']

        # Exit logic
        to_close = []
        for pid, pos in positions.items():
            days = (date - pos['entry_date']).days
            ret = (price - pos['entry_price']) / pos['entry_price']
            if price > pos['high']:
                pos['high'] = price
            if not pos['trail_active'] and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = price * (1 - TRAIL_DISTANCE)
            if pos['trail_active']:
                new_stop = price * (1 - TRAIL_DISTANCE)
                if new_stop > pos['trail_stop']:
                    pos['trail_stop'] = new_stop
            # Exit cond
            if days >= 1 and (
                ret >= PROFIT_TARGET or ret <= -STOP_LOSS or sig == 0 or (pos['trail_active'] and price <= pos['trail_stop'])
            ):
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                charges = calc_charges(buy_val, sell_val)
                exit_val = sell_val * (1 - TRANSACTION_COST)
                pnl = exit_val - buy_val - charges
                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'days_held': days,
                    'pnl': pnl,
                    'return_pct': ret * 100,
                    'exit_reason': (
                        'Profit Target' if ret >= PROFIT_TARGET else
                        'Stop Loss' if ret <= -STOP_LOSS else
                        'Signal Close' if sig == 0 else
                        'Trailing Stop'
                    ),
                    'entry_type': pos['entry_type'],
                    'trade_number': trade_count + 1
                })
                cash += exit_val
                to_close.append(pid)
                trade_count += 1
        for pid in to_close:
            positions.pop(pid)

        if trade_count >= MAX_TRADES:
            break

        # Entry logic
        if sig == 1 and len(positions) < MAX_POSITIONS and cash >= POSITION_SIZE:
            shares = POSITION_SIZE / price
            cost_val = POSITION_SIZE * (1 + TRANSACTION_COST)
            if cash >= cost_val:
                positions[len(positions)+1] = {
                    'entry_date': date,
                    'entry_price': price,
                    'shares': shares,
                    'high': price,
                    'trail_active': False,
                    'trail_stop': 0,
                    'entry_type': sigtype
                }
                cash -= cost_val

    # Final exits
    last_date = df.iloc[-1]['date']
    last_price = df.iloc[-1]['close']
    for pos in positions.values():
        days = (last_date - pos['entry_date']).days
        if days >= 1:
            buy_val = pos['shares'] * pos['entry_price']
            sell_val = pos['shares'] * last_price
            charges = calc_charges(buy_val, sell_val)
            exit_val = sell_val * (1 - TRANSACTION_COST)
            pnl = exit_val - buy_val - charges
            trades.append({
                'symbol': symbol,
                'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                'exit_date': last_date.strftime('%Y-%m-%d'),
                'entry_price': pos['entry_price'],
                'exit_price': last_price,
                'days_held': days,
                'pnl': pnl,
                'return_pct': ((last_price - pos['entry_price']) / pos['entry_price']) * 100,
                'exit_reason': 'EOD Exit',
                'entry_type': pos['entry_type'],
                'trade_number': trade_count + 1
            })
            cash += exit_val
            trade_count += 1

    df_trades = pd.DataFrame(trades)
    total_ret = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    win_rate = (df_trades['pnl'] > 0).mean() * 100 if not df_trades.empty else 0
    avg_hold = df_trades['days_held'].mean() if not df_trades.empty else 0
    sharpe = df_trades['return_pct'].mean() / df_trades['return_pct'].std() if not df_trades.empty and df_trades['return_pct'].std() != 0 else np.nan
    return total_ret, win_rate, avg_hold, sharpe, trade_count, df_trades

# ----- MAIN -----
if __name__ == "__main__":
    print(f"Loading ML model from {MODEL_PATH}..." if USE_ML_CONFIRM else "Using signals only (no ML confirmation)...")
    model = joblib.load(MODEL_PATH) if USE_ML_CONFIRM else None

    all_trade_logs = []
    results_summary = {'return': [], 'win_rate': [], 'avg_hold': [], 'sharpe': [], 'trades': []}

    for symbol in SYMBOLS_TO_TEST:
        filepath = os.path.join(DAILY_DIR, f"{symbol}.csv")
        if not os.path.isfile(filepath):
            print(f"Symbol CSV not found: {filepath}, skipping.")
            continue
        df = pd.read_csv(filepath, parse_dates=['date'])

        # 5-year filter
        cutoff_date = datetime.now() - timedelta(days=5*365)
        df = df[df['date'] >= cutoff_date].reset_index(drop=True)
        if df.empty:
            print(f"No data for last 5 years for {symbol}, skipping.")
            continue

        # Add all indicators
        df = add_indicators(df)
        df = willr_signals(df)
        df = breakout_signals(df)
        df = pullback_signals(df)

        # ML confirmation
        if USE_ML_CONFIRM and model is not None:
            df = ml_signals(df, model)
            # Only allow entry if both (combined signal and ML)
            df = combine_signals(df)
            df['entry_signal'] = np.where((df['entry_signal'] == 1) & (df['ml_signal'] == 1), 1, 0)
        else:
            df = combine_signals(df)

        # Regime filter (trend, volume, ADX)
        df = regime_filter(df)
        if df.empty or 'entry_signal' not in df.columns:
            print(f"No valid signals for {symbol} after filtering, skipping.")
            continue

        ret, winr, avg_hld, shrp, ntrades, trade_logs = backtest(df, symbol)
        print(f"{symbol}: Return={ret:.2f}%, WinRate={winr:.2f}%, Trades={ntrades}")

        results_summary['return'].append(ret)
        results_summary['win_rate'].append(winr)
        results_summary['avg_hold'].append(avg_hld)
        results_summary['sharpe'].append(shrp)
        results_summary['trades'].append(ntrades)
        all_trade_logs.append(trade_logs)

    if all_trade_logs:
        df_all_trades = pd.concat(all_trade_logs, ignore_index=True)
        df_all_trades.to_csv(TRADE_LOG_CSV, index=False)
        print(f"\nFull trade log saved to {TRADE_LOG_CSV}")

    print("\nBacktest Summary Across All Symbols:")
    print(f"Average Return: {np.mean(results_summary['return']):.2f}%")
    print(f"Average Win Rate: {np.mean(results_summary['win_rate']):.2f}%")
    print(f"Average Hold Duration (days): {np.mean(results_summary['avg_hold']):.2f}")
    print(f"Average Sharpe Ratio: {np.nanmean(results_summary['sharpe']):.2f}")
    print(f"Total Trades: {np.sum(results_summary['trades'])}")
