import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# --- Config ---
GOOGLE_SHEET_ID = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
GOOGLE_CREDS_JSON = "falah-credentials.json"
DATA_DIR_DAILY = "/root/falah-ai-bot/swing_data"
YEARS_BACK = 2
INITIAL_CAPITAL = 200_000  # Changed to 2 lakhs
RISK_PER_TRADE = 0.01 * INITIAL_CAPITAL
TRANSACTION_COST = 0.001
MAX_POSITIONS = 5
MAX_TRADES = 2000

# Parameter sweep ranges
SL_ATR_MULT_RANGE = [2.0, 2.5, 2.8, 3.0]
CHANDLER_MULT_RANGE = [2.5, 3.0, 3.5, 4.0]

def get_symbols_from_gsheet(sheet_id, worksheet_name="HalalList"):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_JSON, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)
    symbols = worksheet.col_values(1)
    return [s.strip() for s in symbols if s.strip()]

def load_data(symbol, timeframe="daily"):
    if timeframe == "daily":
        data_dir = DATA_DIR_DAILY
    else:
        raise ValueError("Unsupported timeframe")
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"Warning: Data not found for symbol '{symbol}' in timeframe {timeframe}, skipping.")
        return None
    df = pd.read_csv(path, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365 * YEARS_BACK)
    df = df[df['date'] >= cutoff]
    return df

def compute_indicators(df, chandelier_mult):
    df = df.copy()
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_line'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_lower'] = bbands['BBL_20_2.0']
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['chandelier_exit'] = df['close'] - chandelier_mult * df['atr']
    df['high_1d_ago'] = df['high'].shift(1)
    df['low_1d_ago'] = df['low'].shift(1)
    df['volume_1d_ago'] = df['volume'].shift(1)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def bullish_entry_filter(row):
    return (
        row['macd_line'] > 0 and
        row['macd_signal'] > 0 and
        40 <= row['rsi_14'] <= 70 and
        row['high'] > row['high_1d_ago'] and
        row['low'] > row['low_1d_ago'] and
        row['volume'] > row['volume_1d_ago'] and
        row['close'] >= row['bb_lower']
    )

def exit_logic_chandelier(row):
    return row['close'] < row['chandelier_exit']

def backtest_symbol(df, symbol, sl_atr_mult, chandelier_mult):
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    trade_count = 0
    entry_signal_count = 0
    exit_signal_count = 0
    equity_curve = []
    portfolio_value = cash

    for i in range(1, len(df)):
        row = df.iloc[i]
        date = row['date']
        price = row['close']

        positions_to_close = []
        for pos in positions:
            direction = pos['direction']
            entry_price = pos['entry_price']
            atr = pos['atr']
            stop_loss_price = entry_price - direction * sl_atr_mult * atr
            stop_loss_hit = (direction == 1 and df.iloc[i]['low'] <= stop_loss_price) or \
                            (direction == -1 and df.iloc[i]['high'] >= stop_loss_price)
            momentum_exit = exit_logic_chandelier(row)

            if stop_loss_hit or momentum_exit:
                exit_price = stop_loss_price if stop_loss_hit else price
                pnl = direction * (exit_price - entry_price) * pos['shares']
                pnl -= abs(pnl) * TRANSACTION_COST * 2
                cash += exit_price * pos['shares'] * (1 - TRANSACTION_COST)
                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': 'Long' if direction == 1 else 'Short',
                    'exit_reason': 'Stop Loss' if stop_loss_hit else 'Exit Logic'
                })
                exit_signal_count += 1
                positions_to_close.append(pos)
                trade_count += 1
                if trade_count >= MAX_TRADES:
                    break
        for pos in positions_to_close:
            positions.remove(pos)

        if trade_count >= MAX_TRADES:
            break

        if len(positions) < MAX_POSITIONS and cash > 0:
            if bullish_entry_filter(row):
                atr = row['atr']
                stop_loss_distance = sl_atr_mult * atr
                position_size = RISK_PER_TRADE / stop_loss_distance
                shares = min(cash / price, position_size)
                if shares < 1:
                    continue
                cash -= shares * price * (1 + TRANSACTION_COST)
                pos = {'entry_date': date, 'entry_price': price, 'shares': shares, 'direction': 1, 'atr': atr}
                positions.append(pos)
                entry_signal_count += 1

        # Update portfolio_value for equity curve
        position_value = sum((row['close'] - pos['entry_price']) * pos['shares'] * pos['direction'] for pos in positions)
        portfolio_value = cash + position_value
        equity_curve.append(portfolio_value)

    # Close remaining open positions
    final_date = df.iloc[-1]['date']
    final_close = df.iloc[-1]['close']
    for pos in positions:
        direction = pos['direction']
        entry_price = pos['entry_price']
        shares = pos['shares']
        pnl = direction * (final_close - entry_price) * shares
        pnl -= abs(pnl) * TRANSACTION_COST * 2
        cash += final_close * shares * (1 - TRANSACTION_COST)
        trades.append({
            'symbol': symbol,
            'entry_date': pos['entry_date'],
            'exit_date': final_date,
            'pnl': pnl,
            'entry_price': entry_price,
            'exit_price': final_close,
            'direction': 'Long' if direction == 1 else 'Short',
            'exit_reason': 'EOD Exit'
        })
        equity_curve.append(cash)  # update equity curve with final close

    total_pnl = sum(t['pnl'] for t in trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades else 0

    # Compute max drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (running_max - equity_array) / running_max
    max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0

    # Average trade duration
    trade_durations = []
    # Collect trade durations by date difference
    for t in trades:
        d = (t['exit_date'] - t['entry_date']).days
        if d >= 0:
            trade_durations.append(d)
    avg_duration = np.mean(trade_durations) if trade_durations else 0

    print(f"\n{symbol}: SL ATR {sl_atr_mult}, Chandler Mult {chandelier_mult}")
    print(f"Trades: {total_trades}, Entries: {entry_signal_count}, Exits (excl stop loss): {exit_signal_count}")
    print(f"Total PnL: {total_pnl:.2f}, Win Rate: {win_rate:.2f}%, Max Drawdown: {max_drawdown:.2f}%, Avg Trade Duration: {avg_duration:.2f} days")

    return total_pnl, win_rate, max_drawdown, avg_duration

def parameter_sweep():
    symbols = get_symbols_from_gsheet(GOOGLE_SHEET_ID)
    results = []

    for sl_atr_mult in SL_ATR_MULT_RANGE:
        for chandelier_mult in CHANDLER_MULT_RANGE:
            total_pnl_sum = 0
            total_wins_sum = 0
            total_trades_sum = 0
            total_drawdown_list = []
            total_duration_list = []
            for symbol in symbols:
                daily_df = load_data(symbol, "daily")
                if daily_df is None or len(daily_df) < 20:
                    continue
                df = compute_indicators(daily_df, chandelier_mult)
                pnl, win_rate, max_dd, avg_dur = backtest_symbol(df, symbol, sl_atr_mult, chandelier_mult)
                total_pnl_sum += pnl
                total_drawdown_list.append(max_dd)
                total_duration_list.append(avg_dur)
            avg_drawdown = np.mean(total_drawdown_list) if total_drawdown_list else 0
            avg_duration = np.mean(total_duration_list) if total_duration_list else 0
            results.append({
                'SL_ATR_MULT': sl_atr_mult,
                'CHANDLER_MULT': chandelier_mult,
                'TOTAL_PnL': total_pnl_sum,
                'AVG_MAX_DD_pct': avg_drawdown,
                'AVG_TRADE_DURATION_days': avg_duration
            })

    # Sort by total PnL descending
    results = sorted(results, key=lambda x: x['TOTAL_PnL'], reverse=True)

    print("\n=== Parameter Sweep Results ===")
    print("SL_ATR_MULT | CHANDLER_MULT | TOTAL_PnL | AVG_MAX_DD% | AVG_TRADE_DURATION_days")
    for r in results:
        print(f"{r['SL_ATR_MULT']:11} | {r['CHANDLER_MULT']:13} | {r['TOTAL_PnL']:9.2f} | {r['AVG_MAX_DD_pct']:10.2f} | {r['AVG_TRADE_DURATION_days']:21.2f}")

if __name__ == "__main__":
    parameter_sweep()
