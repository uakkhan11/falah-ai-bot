import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from strategy_utils import add_indicators, combine_signals

# Configure your data directories as per your improved_fetcher.py setup
DATA_DIR_DAILY = "/root/falah-ai-bot/swing_data"
DATA_DIR_1H = "/root/falah-ai-bot/intraday_swing_data"
DATA_DIR_15M = "/root/falah-ai-bot/scalping_data"

SYMBOLS = ["RELIANCE", "SUNPHARMA"]
YEARS_BACK = 2

def add_weekly_ema(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df_weekly = df.set_index('date').resample('W-MON')['close'].last().dropna().to_frame()
    df_weekly['ema50'] = ta.ema(df_weekly['close'], length=50)
    df_weekly['ema50_slope'] = df_weekly['ema50'].diff()
    df = df.set_index('date')
    df['weekly_ema50'] = df_weekly['ema50'].reindex(df.index, method='ffill')
    df['weekly_ema50_slope'] = df_weekly['ema50_slope'].reindex(df.index, method='ffill')
    df = df.reset_index()
    return df

def load_candle_data(symbol, timeframe, years=YEARS_BACK):
    folder_map = {
        "daily": DATA_DIR_DAILY,
        "1h": DATA_DIR_1H,
        "15m": DATA_DIR_15M,
    }
    path = os.path.join(folder_map[timeframe], f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"Data file missing for {symbol} timeframe {timeframe}: {path}")
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    cutoff = datetime.now() - timedelta(days=365 * years)
    df = df[df['date'] >= cutoff].sort_values('date').reset_index(drop=True)
    return df

def prepare_multitimeframe_data(symbol):
    df_daily = load_candle_data(symbol, "daily")
    df_1h = load_candle_data(symbol, "1h")
    df_15m = load_candle_data(symbol, "15m")
    if df_daily is None or df_1h is None or df_15m is None:
        return None

    df_15m['rsi_15m'] = ta.rsi(df_15m['close'], length=14)
    df_15m['ema20_15m'] = ta.ema(df_15m['close'], length=20)
    df_15m['ema20_15m_slope'] = df_15m['ema20_15m'].diff()

    df_1h['rsi_1h'] = ta.rsi(df_1h['close'], length=14)
    df_1h['ema50_1h'] = ta.ema(df_1h['close'], length=50)
    df_1h['ema50_1h_slope'] = df_1h['ema50_1h'].diff()

    daily_15m_agg = df_15m.groupby(df_15m['date'].dt.floor('D')).agg({
        'rsi_15m': 'last',
        'ema20_15m_slope': 'last'
    }).rename_axis('date').reset_index()

    daily_1h_agg = df_1h.groupby(df_1h['date'].dt.floor('D')).agg({
        'rsi_1h': 'last',
        'ema50_1h_slope': 'last'
    }).rename_axis('date').reset_index()

    df_daily = pd.merge(df_daily, daily_15m_agg, on='date', how='left')
    df_daily = pd.merge(df_daily, daily_1h_agg, on='date', how='left')

    # Proper fillna avoiding pandas chained assignment warning
    df_daily['rsi_15m'] = df_daily['rsi_15m'].ffill()
    df_daily['ema20_15m_slope'] = df_daily['ema20_15m_slope'].fillna(0)
    df_daily['rsi_1h'] = df_daily['rsi_1h'].ffill()
    df_daily['ema50_1h_slope'] = df_daily['ema50_1h_slope'].fillna(0)

    df_daily = add_indicators(df_daily)
    df_daily = add_weekly_ema(df_daily)

    return df_daily

def modify_combine_signals_with_mtf(df):
    df = combine_signals(df)
    long_cond = (
        (df['entry_signal'] == 1) &
        (df['rsi_15m'] > 50) &
        (df['rsi_1h'] > 50) &
        (df['ema20_15m_slope'] > 0) &
        (df['ema50_1h_slope'] > 0)
    )
    short_cond = (
        (df['close'] < df['ema200']) &
        (df['rsi_15m'] < 50) &
        (df['rsi_1h'] < 50) &
        (df['ema20_15m_slope'] < 0) &
        (df['ema50_1h_slope'] < 0)
    )
    df['entry_signal_long'] = long_cond
    df['entry_signal_short'] = short_cond
    df['entry_signal_final'] = 0
    df.loc[long_cond, 'entry_signal_final'] = 1
    df.loc[short_cond, 'entry_signal_final'] = -1
    return df

def backtest_mtf(df, symbol):
    INITIAL_CAPITAL = 1_000_000
    RISK_PER_TRADE = 0.01 * INITIAL_CAPITAL
    PROFIT_TARGET1 = 0.10
    PROFIT_TARGET2 = 0.15
    ATR_SL_MULT = 2.8
    TRAIL_TRIGGER = 0.07
    MAX_POSITIONS = 5
    MAX_TRADES = 2000
    TRANSACTION_COST = 0.001
    RSI_THRESHOLD = 55
    EMA_SLOPE_THRESHOLD = 0.0
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0
    regime_fail_count = {}

    rolling_atr_mean = df['atr'].rolling(window=20, min_periods=1).mean()

    for i in range(1, len(df)):
        row = df.iloc[i]
        date, price = row['date'], row['close']
        sig = row.get('entry_signal_final', 0)
        sigtype = 'Long' if sig == 1 else ('Short' if sig == -1 else '')

        regime_ok = (
            (df.at[i, 'ema200_slope'] > EMA_SLOPE_THRESHOLD) and
            (row['rsi'] > RSI_THRESHOLD)
        )

        if i > 0 and not pd.isna(row['atr']):
            stop_loss_distance = ATR_SL_MULT * row['atr']
        else:
            stop_loss_distance = ATR_SL_MULT * (df['atr'].mean() if not df['atr'].isna().all() else 1)

        position_size = min(cash, RISK_PER_TRADE / stop_loss_distance * price)

        to_close = []
        for pid, pos in list(positions.items()):
            direction = pos.get('direction', 1)
            ret = direction * (price - pos['entry_price']) / pos['entry_price']

            adaptive_atr_mult = ATR_SL_MULT * (rolling_atr_mean.iloc[i] / rolling_atr_mean.mean())
            adaptive_stop_loss = pos['entry_price'] - direction * adaptive_atr_mult * pos.get('entry_atr', 0)

            if (direction == 1 and price > pos['high']) or (direction == -1 and price < pos['low']):
                if direction == 1:
                    pos['high'] = price
                else:
                    pos['low'] = price

            if not pos.get('trail_active', False) and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = row.get('chandelier_exit', 0)

            if pos.get('trail_active', False):
                if (direction == 1 and row.get('chandelier_exit', 0) > pos.get('trail_stop', 0)) or \
                   (direction == -1 and row.get('chandelier_exit', 0) < pos.get('trail_stop', 0)):
                    pos['trail_stop'] = row.get('chandelier_exit', 0)

            reason = None
            pnl = 0

            # Implement your partial scaling and exit logic here
            # Example for partial with placeholders, extend as per your rules

            if ret >= PROFIT_TARGET1 and not pos.get('scale1', False):
                # Partial profit taking logic here
                pass
            elif ret >= PROFIT_TARGET2 and not pos.get('scale2', False):
                # Second partial profit taking / exit
                pass
            elif (direction == 1 and price <= adaptive_stop_loss) or (direction == -1 and price >= adaptive_stop_loss):
                reason = 'ATR Stop Loss'
            elif pos.get('trail_active', False) and (
                (direction == 1 and price <= pos.get('trail_stop', 0)) or (direction == -1 and price >= pos.get('trail_stop', 0))
            ):
                reason = 'Trailing Stop'
            elif (direction == 1 and price <= pos.get('chandelier_exit', 0)) or (direction == -1 and price >= pos.get('chandelier_exit', 0)):
                reason = 'Chandelier Exit'
            else:
                pid_key = f"{symbol}_{pid}"
                if not regime_ok:
                    regime_fail_count[pid_key] = regime_fail_count.get(pid_key, 0) + 1
                else:
                    regime_fail_count[pid_key] = 0
                if regime_fail_count.get(pid_key, 0) >= 2:
                    reason = 'Regime Exit'

            if reason:
                buy_val = pos['shares'] * pos['entry_price']
                sell_val = pos['shares'] * price
                charges = (buy_val + sell_val) * TRANSACTION_COST
                pnl = direction * (sell_val * (1 - TRANSACTION_COST) - buy_val) - charges

                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_type': pos['entry_type'],
                    'exit_reason': reason
                })

                cash += sell_val
                to_close.append(pid)
                trade_count += 1

                if trade_count >= MAX_TRADES:
                    break

        for pid in to_close:
            if pid in positions:
                del positions[pid]

        if trade_count >= MAX_TRADES:
            break

        if sig != 0 and len(positions) < MAX_POSITIONS and cash >= position_size:
            shares = position_size / price
            direction = 1 if sig == 1 else -1
            positions[len(positions) + 1] = {
                'entry_date': date,
                'entry_price': price,
                'shares': shares,
                'high': price,
                'low': price,
                'trail_active': False,
                'trail_stop': 0,
                'entry_atr': row.get('atr', 0),
                'entry_type': sigtype,
                'scale1': False,
                'scale2': False,
                'direction': direction,
            }
            cash -= position_size * (1 + TRANSACTION_COST)

    return trades

def main():
    all_trades = []
    for symbol in SYMBOLS:
        print(f"\nBacktesting {symbol} multi-timeframe for last {YEARS_BACK} years...")
        df = prepare_multitimeframe_data(symbol)
        if df is None:
            print(f"Skipping {symbol} due to missing data.")
            continue
        df = modify_combine_signals_with_mtf(df)
        trades = backtest_mtf(df, symbol)
        all_trades.extend(trades)

    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        print(f"\nTotal trades executed: {len(trades_df)}")
        print(f"Total PnL: {trades_df['pnl'].sum():.2f}")
        print(f"Win rate: {(trades_df['pnl'] > 0).mean() * 100:.2f}%")
        print("\nPnL by Entry Type:")
        print(trades_df.groupby('entry_type')['pnl'].agg(['count','sum','mean']))
        print("\nExit Reason Performance:")
        print(trades_df.groupby('exit_reason')['pnl'].agg(['count','sum','mean']))
    else:
        print("No trades executed.")

if __name__ == "__main__":
    main()
