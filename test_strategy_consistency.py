import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from improved_fetcher import SmartHalalFetcher
from strategy_utils import add_indicators, combine_signals

DATA_DIR_DAILY = "swing_data/daily"
DATA_DIR_1H = "swing_data/1hour"
DATA_DIR_15M = "swing_data/15minute"
SYMBOLS = ["RELIANCE", "SUNPHARMA"]
YEARS_BACK = 2

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

    # Calculate intraday indicators
    df_15m['rsi_15m'] = ta.rsi(df_15m['close'], length=14)
    df_15m['ema20_15m'] = ta.ema(df_15m['close'], length=20)
    df_15m['ema20_15m_slope'] = df_15m['ema20_15m'].diff()

    df_1h['rsi_1h'] = ta.rsi(df_1h['close'], length=14)
    df_1h['ema50_1h'] = ta.ema(df_1h['close'], length=50)
    df_1h['ema50_1h_slope'] = df_1h['ema50_1h'].diff()

    # Aggregate intraday data to daily level
    daily_15m_agg = df_15m.groupby(df_15m['date'].dt.floor('D')).agg({
        'rsi_15m': 'last',
        'ema20_15m_slope': 'last'
    }).rename_axis('date').reset_index()

    daily_1h_agg = df_1h.groupby(df_1h['date'].dt.floor('D')).agg({
        'rsi_1h': 'last',
        'ema50_1h_slope': 'last'
    }).rename_axis('date').reset_index()

    # Merge into daily df
    df_daily = pd.merge(df_daily, daily_15m_agg, on='date', how='left')
    df_daily = pd.merge(df_daily, daily_1h_agg, on='date', how='left')

    # Fill missing values
    df_daily['rsi_15m'].fillna(method='ffill', inplace=True)
    df_daily['ema20_15m_slope'].fillna(0, inplace=True)
    df_daily['rsi_1h'].fillna(method='ffill', inplace=True)
    df_daily['ema50_1h_slope'].fillna(0, inplace=True)

    # Add daily indicators
    df_daily = add_indicators(df_daily)
    df_daily = add_weekly_ema(df_daily)

    return df_daily

def modify_combine_signals_with_mtf(df):
    # Base daily signals
    df = combine_signals(df)

    # Multi-timeframe confirmations
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

        # Regime check based on trend and thresholds
        regime_ok = (
            (df.at[i, 'ema200_slope'] > EMA_SLOPE_THRESHOLD) and
            (row['rsi'] > RSI_THRESHOLD)
        )

        if i > 0 and not pd.isna(row['atr']):
            stop_loss_distance = ATR_SL_MULT * row['atr']
        else:
            stop_loss_distance = ATR_SL_MULT * (df['atr'].mean() if not df['atr'].isna().all() else 1)

        position_size = min(cash, RISK_PER_TRADE / stop_loss_distance * price)

        # EXIT & ENTRY logic extended for both long and short trades
        to_close = []
        for pid, pos in list(positions.items()):
            direction = pos.get('direction', 1)  # 1 for long, -1 for short
            ret = direction * (price - pos['entry_price']) / pos['entry_price']

            adaptive_atr_mult = ATR_SL_MULT * (rolling_atr_mean.iloc[i] / rolling_atr_mean.mean())
            adaptive_stop_loss = pos['entry_price'] - direction * adaptive_atr_mult * pos.get('entry_atr', 0)

            if (direction == 1 and price > pos['high']) or (direction == -1 and price < pos['low']):
                pos['high' if direction == 1 else 'low'] = price

            if not pos.get('trail_active', False) and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = row.get('chandelier_exit', 0)

            if pos.get('trail_active', False):
                if (direction == 1 and row.get('chandelier_exit', 0) > pos.get('trail_stop', 0)) or \
                   (direction == -1 and row.get('chandelier_exit', 0) < pos.get('trail_stop', 0)):
                    pos['trail_stop'] = row.get('chandelier_exit', 0)

            reason = None
            pnl = 0

            # Partial scaling logic same as before with consideration for direction
            # Profit targets and stop losses adjusted for direction

            # ... implement exits, partial scaling, and regime exits ...

            # For brevity, please extend as per long-only version with sign adjustments

        # Cleanup closed positions
        for pid in to_close:
            if pid in positions:
                del positions[pid]

        if trade_count >= MAX_TRADES:
            break

        # ENTRY - support long and short trade entries
        if sig != 0 and len(positions) < MAX_POSITIONS and cash >= position_size:
            shares = position_size / price
            direction = 1 if sig == 1 else -1
            positions[len(positions) + 1] = {
                'entry_date': date,
                'entry_price': price,
                'shares': shares,
                'high': price,
                'low': price,  # For shorts track low instead of high
                'trail_active': False,
                'trail_stop': 0,
                'entry_atr': row.get('atr', 0),
                'entry_type': sigtype,
                'scaled_out': False,
                'scale1': False,
                'scale2': False,
                'direction': direction,
            }
            cash -= position_size * (1 + TRANSACTION_COST)

    return trades

def main():
    all_trades = []
    for symbol in SYMBOLS:
        print(f"\nBacktesting {symbol} with multi-timeframe for last {YEARS_BACK} years...")
        df = prepare_multitimeframe_data(symbol)
        if df is None:
            continue
        df = modify_combine_signals_with_mtf(df)
        trades = backtest_mtf(df, symbol)
        all_trades.extend(trades)
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        total_trades = len(trades_df)
        total_pnl = trades_df['pnl'].sum()
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        print(f"\nTotal trades executed: {total_trades}")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Win rate: {win_rate:.2f}%")
        print("\nPnL by Entry Type:")
        print(trades_df.groupby('entry_type')['pnl'].agg(['count','sum','mean']))
        print("\nExit Reason Performance:")
        print(trades_df.groupby('exit_reason')['pnl'].agg(['count','sum','mean']))
    else:
        print("No trades executed during backtest period.")

if __name__ == "__main__":
    main()
