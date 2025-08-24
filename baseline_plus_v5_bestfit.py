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
DATA_DIR_INTRADAY_1H = "/root/falah-ai-bot/intraday_swing_data"
DATA_DIR_INTRADAY_15M = "/root/falah-ai-bot/scalping_data"
YEARS_BACK = 2
SL_ATR_MULT = 2.8
INITIAL_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.01 * INITIAL_CAPITAL
TRANSACTION_COST = 0.001
MAX_POSITIONS = 5
MAX_TRADES = 2000

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
    elif timeframe == "1h":
        data_dir = DATA_DIR_INTRADAY_1H
    elif timeframe == "15m":
        data_dir = DATA_DIR_INTRADAY_15M
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

def compute_indicators(df):
    df = df.copy()
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_line'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_lower'] = bbands['BBL_20_2.0']
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['chandelier_exit'] = df['close'] - 3 * df['atr']
    supertrend_df = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
    df['supertrend_direction'] = supertrend_df['SUPERTd_10_3.0']
    df['high_1d_ago'] = df['high'].shift(1)
    df['low_1d_ago'] = df['low'].shift(1)
    df['volume_1d_ago'] = df['volume'].shift(1)
    df = calculate_fib_retracement(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_fib_retracement(df):
    # Calculate daily Fibonacci retracement levels using rolling window
    # Using last 14 days high and low for calculation
    period = 14
    df['fib_high'] = df['high'].rolling(period).max()
    df['fib_low'] = df['low'].rolling(period).min()
    df['fib_382'] = df['fib_high'] - 0.382 * (df['fib_high'] - df['fib_low'])
    df['fib_618'] = df['fib_high'] - 0.618 * (df['fib_high'] - df['fib_low'])
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

# Exit Case 1: Chandelier exit only (stop loss handled in backtest)
def exit_case_1(row):
    return row['close'] < row['chandelier_exit']

# Exit Case 2: Multi-timeframe not needed in exit logic, same as case 1 (chandelier)
# But data includes intraday merged features applied before backtest

# Exit Case 3: Fibonacci retracement exit with ATR stop loss
def exit_case_3(row):
    # Exit if price falls below 61.8% fib retracement level
    below_fib_618 = row['close'] < row['fib_618']
    return below_fib_618

def merge_multitimeframe(daily_df, df_1h, df_15m):
    # Merge 1h and 15m aggregated features as daily max/min/close/volume
    df = daily_df.copy()
    if df_1h is not None:
        df_1h = df_1h.copy()
        df_1h['date_daily'] = df_1h['date'].dt.floor('D')
        agg_1h = df_1h.groupby('date_daily').agg({
            'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).rename(columns=lambda x: x + '_1h')
        df = df.merge(agg_1h, how='left', left_on='date', right_on='date_daily')
        if 'date_daily' in df.columns:
            df.drop(columns=['date_daily'], inplace=True)
    if df_15m is not None:
        df_15m = df_15m.copy()
        df_15m['date_daily'] = df_15m['date'].dt.floor('D')
        agg_15m = df_15m.groupby('date_daily').agg({
            'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).rename(columns=lambda x: x + '_15m')
        df = df.merge(agg_15m, how='left', left_on='date', right_on='date_daily')
        if 'date_daily' in df.columns:
            df.drop(columns=['date_daily'], inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

def backtest_symbol(df, symbol, exit_logic, label):
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    trade_count = 0
    entry_signal_count = 0
    exit_signal_count = 0
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        date = row['date']
        price = row['close']

        positions_to_close = []
        for pos in positions:
            direction = pos['direction']
            entry_price = pos['entry_price']
            atr = pos['atr']
            stop_loss_price = entry_price - direction * SL_ATR_MULT * atr
            stop_loss_hit = (direction == 1 and df.iloc[i]['low'] <= stop_loss_price) or \
                            (direction == -1 and df.iloc[i]['high'] >= stop_loss_price)
            momentum_exit = exit_logic(row)

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
                stop_loss_distance = SL_ATR_MULT * atr
                position_size = RISK_PER_TRADE / stop_loss_distance
                shares = min(cash / price, position_size)
                if shares < 1:
                    continue
                cash -= shares * price * (1 + TRANSACTION_COST)
                pos = {'entry_date': date, 'entry_price': price, 'shares': shares, 'direction': 1, 'atr': atr}
                positions.append(pos)
                entry_signal_count += 1
        if trade_count >= MAX_TRADES:
            break

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

    total_pnl = sum(t['pnl'] for t in trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades else 0

    print(f"\n{label} - {symbol} Summary:")
    print(f"Entry signals taken: {entry_signal_count}")
    print(f"Total trades executed: {total_trades}")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Exit signals triggered (excluding stop loss): {exit_signal_count}")

    return trades, entry_signal_count, exit_signal_count

def main():
    symbols = get_symbols_from_gsheet(GOOGLE_SHEET_ID)

    # Case 1: Daily only
    print("=== Running Case 1: Daily Data with Chandelier Exit + ATR SL ===")
    summary_case_1 = {'trades': [], 'entries': 0, 'exits': 0}
    for symbol in symbols:
        daily_df = load_data(symbol, "daily")
        if daily_df is None or len(daily_df) < 20:
            continue
        daily_df = compute_indicators(daily_df)
        trades, entries, exits = backtest_symbol(daily_df, symbol, exit_case_1, "Case 1")
        summary_case_1['trades'].extend(trades)
        summary_case_1['entries'] += entries
        summary_case_1['exits'] += exits

    print("\nCase 1 Overall Summary:")
    print(f"Total trades: {len(summary_case_1['trades'])}")
    print(f"Total entries signaled: {summary_case_1['entries']}")
    print(f"Total exit logic triggered (excl. stop loss): {summary_case_1['exits']}")
    print(f"Total PnL: {sum(t['pnl'] for t in summary_case_1['trades']):.2f}")
    wins = sum(1 for t in summary_case_1['trades'] if t['pnl'] > 0)
    win_rate = (wins / len(summary_case_1['trades']) * 100) if summary_case_1['trades'] else 0
    print(f"Win rate: {win_rate:.2f}%")

    # Case 2: Multi-timeframe (daily + 1h + 15m)
    print("\n=== Running Case 2: Multi-timeframe (Daily + 1h + 15m) with Chandelier Exit + ATR SL ===")
    summary_case_2 = {'trades': [], 'entries': 0, 'exits': 0}
    for symbol in symbols:
        daily_df = load_data(symbol, "daily")
        df_1h = load_data(symbol, "1h")
        df_15m = load_data(symbol, "15m")
        if daily_df is None or len(daily_df) < 20:
            continue
        df_merged = merge_multitimeframe(daily_df, df_1h, df_15m)
        df_merged = compute_indicators(df_merged)
        trades, entries, exits = backtest_symbol(df_merged, symbol, exit_case_1, "Case 2")
        summary_case_2['trades'].extend(trades)
        summary_case_2['entries'] += entries
        summary_case_2['exits'] += exits

    print("\nCase 2 Overall Summary:")
    print(f"Total trades: {len(summary_case_2['trades'])}")
    print(f"Total entries signaled: {summary_case_2['entries']}")
    print(f"Total exit logic triggered (excl. stop loss): {summary_case_2['exits']}")
    print(f"Total PnL: {sum(t['pnl'] for t in summary_case_2['trades']):.2f}")
    wins = sum(1 for t in summary_case_2['trades'] if t['pnl'] > 0)
    win_rate = (wins / len(summary_case_2['trades']) * 100) if summary_case_2['trades'] else 0
    print(f"Win rate: {win_rate:.2f}%")

if __name__ == "__main__":
    main()
