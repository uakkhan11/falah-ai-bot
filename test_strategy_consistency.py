import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from strategy_utils import (
    add_indicators,
    combine_signals
)

DATA_DIR = "swing_data"
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

def load_candle_data(symbol, years=YEARS_BACK):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"Data file missing for {symbol}: {path}")
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    cutoff = datetime.now() - timedelta(days=365 * years)
    df = df[df['date'] >= cutoff].sort_values('date').reset_index(drop=True)
    return df

def backtest(df, symbol):
    INITIAL_CAPITAL = 1_000_000
    RISK_PER_TRADE = 0.01 * INITIAL_CAPITAL  # 1% risk per trade
    PROFIT_TARGET1 = 0.10
    PROFIT_TARGET2 = 0.15
    ATR_SL_MULT = 2.8
    TRAIL_TRIGGER = 0.07
    MAX_POSITIONS = 5
    MAX_TRADES = 2000
    TRANSACTION_COST = 0.001  # 0.1%

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
        sig, sigtype = row.get('entry_signal', 0), row.get('entry_type', '')

        regime_ok = (
            (price > row['ema200']) and
            (row['adx'] > 15) and
            (df.at[i, 'ema200_slope'] > EMA_SLOPE_THRESHOLD) and
            (row['rsi'] > RSI_THRESHOLD) and
            (df.at[i, 'weekly_ema50_slope'] > EMA_SLOPE_THRESHOLD)
        )

        if i > 0 and not pd.isna(row['atr']):
            stop_loss_distance = ATR_SL_MULT * row['atr']
        else:
            stop_loss_distance = ATR_SL_MULT * (df['atr'].mean() if not df['atr'].isna().all() else 1)

        position_size = min(cash, RISK_PER_TRADE / stop_loss_distance * price)

        to_close = []
        for pid, pos in list(positions.items()):
            ret = (price - pos['entry_price']) / pos['entry_price']

            adaptive_atr_mult = ATR_SL_MULT * (rolling_atr_mean.iloc[i] / rolling_atr_mean.mean())
            adaptive_stop_loss = pos['entry_price'] - adaptive_atr_mult * pos.get('entry_atr', 0)

            if price > pos['high']:
                pos['high'] = price

            if not pos.get('trail_active', False) and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = row.get('chandelier_exit', 0)

            if pos.get('trail_active', False) and row.get('chandelier_exit', 0) > pos.get('trail_stop', 0):
                pos['trail_stop'] = row.get('chandelier_exit', 0)

            reason = None
            pnl = 0

            if ret >= PROFIT_TARGET1 and 'scale1' not in pos:
                scale_qty = pos['shares'] * 0.5
                remain_qty = pos['shares'] - scale_qty
                buy_val = scale_qty * pos['entry_price']
                sell_val = scale_qty * price
                charges = (buy_val + sell_val) * TRANSACTION_COST
                pnl = sell_val * (1 - TRANSACTION_COST) - buy_val - charges

                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_type': pos['entry_type'],
                    'exit_reason': 'Partial Profit Target 1'
                })

                pos['shares'] = remain_qty
                pos['scale1'] = True
                cash += sell_val
                continue

            if ret >= PROFIT_TARGET2 and 'scale2' not in pos:
                scale_qty = pos['shares'] * 0.5
                remain_qty = pos['shares'] - scale_qty
                buy_val = scale_qty * pos['entry_price']
                sell_val = scale_qty * price
                charges = (buy_val + sell_val) * TRANSACTION_COST
                pnl = sell_val * (1 - TRANSACTION_COST) - buy_val - charges

                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_type': pos['entry_type'],
                    'exit_reason': 'Partial Profit Target 2'
                })

                pos['shares'] = remain_qty
                pos['scale2'] = True
                cash += sell_val
                if remain_qty <= 0:
                    to_close.append(pid)
                    trade_count += 1
                continue

            if pos.get('scale2', False):
                reason = 'Profit Target'
            elif price <= adaptive_stop_loss:
                reason = 'ATR Stop Loss'
            elif pos.get('trail_active', False) and price <= pos.get('trail_stop', 0):
                reason = 'Trailing Stop'
            elif price <= pos.get('chandelier_exit', 0):
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
                pnl = sell_val * (1 - TRANSACTION_COST) - buy_val - charges

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

        # ENTRY LOGIC
        if sig == 1 and len(positions) < MAX_POSITIONS and cash >= position_size:
            shares = position_size / price
            positions[len(positions) + 1] = {
                'entry_date': date,
                'entry_price': price,
                'shares': shares,
                'high': price,
                'trail_active': False,
                'trail_stop': 0,
                'entry_atr': row.get('atr', 0),
                'entry_type': sigtype,
                'scale1': False,
                'scale2': False,
            }
            cash -= position_size * (1 + TRANSACTION_COST)

    return trades


def main():
    all_trades = []
    for symbol in SYMBOLS:
        print(f"\nBacktesting {symbol} for last {YEARS_BACK} years...")
        df = load_candle_data(symbol)
        if df is None:
            continue

        df = add_indicators(df)
        df = add_weekly_ema(df)
        df = combine_signals(df)

        trades = backtest(df, symbol)
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
