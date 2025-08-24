import os
import pandas as pd
from datetime import datetime, timedelta
from strategy_utils import (
    add_indicators,
    combine_signals
)

DATA_DIR = "swing_data"
SYMBOLS = ["RELIANCE", "SUNPHARMA"]
YEARS_BACK = 2

def load_candle_data(symbol, years=YEARS_BACK):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"Data file missing for {symbol}: {path}")
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    cutoff = datetime.now() - timedelta(days=365*years)
    df = df[df['date'] >= cutoff].sort_values('date').reset_index(drop=True)
    return df

def backtest(df, symbol):
    INITIAL_CAPITAL = 1_000_000
    POSITION_SIZE = 100_000
    PROFIT_TARGET = 0.10
    ATR_SL_MULT = 2.8
    TRAIL_TRIGGER = 0.07
    MAX_POSITIONS = 5
    MAX_TRADES = 2000
    TRANSACTION_COST = 0.001  # 0.1%
    SCALE_OUT_PCT = 0.5  # Take half profit at target, trail rest

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    trade_count = 0
    regime_fail_count = {}

    for i in range(1, len(df)):
        row = df.iloc[i]
        date, price = row['date'], row['close']
        sig, sigtype = row.get('entry_signal', 0), row.get('entry_type', '')

        regime_ok = (
            (price > row['ema200']) and
            (row['adx'] > 15) and
            (df.at[i, 'ema200_slope'] > 0) and
            (row['rsi'] > 50) and
            (df.at[i, 'weekly_ema50_slope'] > 0)
        )

        # EXIT LOGIC
        to_close = []
        for pid, pos in list(positions.items()):
            ret = (price - pos['entry_price']) / pos['entry_price']
            atr_stop = pos['entry_price'] - ATR_SL_MULT * pos.get('entry_atr', 0)

            if price > pos['high']:
                pos['high'] = price

            if not pos.get('trail_active', False) and ret >= TRAIL_TRIGGER:
                pos['trail_active'] = True
                pos['trail_stop'] = row.get('chandelier_exit', 0)

            if pos.get('trail_active', False) and row.get('chandelier_exit', 0) > pos.get('trail_stop', 0):
                pos['trail_stop'] = row.get('chandelier_exit', 0)

            sell_shares = pos['shares']
            reason = None
            pnl = 0

            # Partial scaling out
            if ret >= PROFIT_TARGET and not pos.get('scaled_out', False):
                scale_qty = pos['shares'] * SCALE_OUT_PCT
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
                    'exit_reason': 'Partial Profit Target'
                })

                pos['shares'] = remain_qty
                pos['scaled_out'] = True
                cash += sell_val
                continue  # keep position with remaining shares

            if ret >= PROFIT_TARGET and pos.get('scaled_out', False):
                reason = 'Profit Target'
            elif price <= atr_stop:
                reason = 'ATR Stop Loss'
            elif pos.get('trail_active', False) and price <= pos.get('trail_stop', 0):
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
            del positions[pid]

        if trade_count >= MAX_TRADES:
            break

        # ENTRY LOGIC
        if sig == 1 and len(positions) < MAX_POSITIONS and cash >= POSITION_SIZE:
            shares = POSITION_SIZE / price
            positions[len(positions) + 1] = {
                'entry_date': date,
                'entry_price': price,
                'shares': shares,
                'high': price,
                'trail_active': False,
                'trail_stop': 0,
                'entry_atr': row.get('atr', 0),
                'entry_type': sigtype,
                'scaled_out': False,
            }
            cash -= POSITION_SIZE * (1 + TRANSACTION_COST)

    return trades

def main():
    all_trades = []
    for symbol in SYMBOLS:
        print(f"\nBacktesting {symbol} for last {YEARS_BACK} years...")
        df = load_candle_data(symbol)
        if df is None:
            continue

        df = add_indicators(df)
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
