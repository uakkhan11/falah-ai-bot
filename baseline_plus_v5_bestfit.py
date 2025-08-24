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
SL_ATR_MULT = 2.8
INITIAL_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.01 * INITIAL_CAPITAL
TRANSACTION_COST = 0.001
MAX_POSITIONS = 5
MAX_TRADES = 2000

def get_symbols_from_gsheet(sheet_id, worksheet_name="HalalList"):
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_JSON, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)
    symbols = worksheet.col_values(1)
    return [s.strip() for s in symbols if s.strip()]

def load_data(symbol):
    path = os.path.join(DATA_DIR_DAILY, f"{symbol}.csv")
    if not os.path.exists(path):
        print(f"Warning: Data not found for symbol '{symbol}', skipping.")
        return None
    df = pd.read_csv(path, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365*YEARS_BACK)
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
    # Calculate chandelier exit as close - ATR * multiplier (e.g., 3)
    df['chandelier_exit'] = df['close'] - 3 * df['atr']

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

def exit_case_1(row):
    return (row['rsi_14'] < 70) and (row['close'] < row['bb_upper'])

def exit_case_2(row):
    return ((row['rsi_14'] < 70) and (row['close'] < row['bb_upper'])) or (row['close'] < row['chandelier_exit'])

def backtest_symbol(df, symbol, exit_logic):
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    trade_count = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        date = row['date']
        price = row['close']

        positions_to_close = []
        for pos in positions:
            # ATR stop loss price
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
                trades.append({'symbol': symbol,
                               'entry_date': pos['entry_date'],
                               'exit_date': date,
                               'pnl': pnl,
                               'entry_price': entry_price,
                               'exit_price': exit_price,
                               'direction': 'Long' if direction == 1 else 'Short',
                               'exit_reason': 'Stop Loss' if stop_loss_hit else 'Exit Logic'})
                positions_to_close.append(pos)
                trade_count += 1
                if trade_count >= MAX_TRADES:
                    break
        for pos in positions_to_close:
            positions.remove(pos)
        if trade_count >= MAX_TRADES:
            break

        # Entry logic
        if len(positions) < MAX_POSITIONS and cash > 0:
            if bullish_entry_filter(row):
                atr = row['atr']
                stop_loss_distance = SL_ATR_MULT * atr
                position_size = RISK_PER_TRADE / stop_loss_distance
                shares = min(cash / price, position_size)
                if shares < 1:
                    continue
                cash -= shares * price * (1 + TRANSACTION_COST)
                pos = {'entry_date': date, 'entry_price': price, 'shares': shares,
                       'direction': 1, 'atr': atr}
                positions.append(pos)
        if trade_count >= MAX_TRADES:
            break

    # Close all open positions at last day close
    final_date = df.iloc[-1]['date']
    final_close = df.iloc[-1]['close']
    for pos in positions:
        direction = pos['direction']
        entry_price = pos['entry_price']
        shares = pos['shares']
        pnl = direction * (final_close - entry_price) * shares
        pnl -= abs(pnl) * TRANSACTION_COST * 2
        cash += final_close * shares * (1 - TRANSACTION_COST)
        trades.append({'symbol': symbol,
                       'entry_date': pos['entry_date'],
                       'exit_date': final_date,
                       'pnl': pnl,
                       'entry_price': entry_price,
                       'exit_price': final_close,
                       'direction': 'Long' if direction == 1 else 'Short',
                       'exit_reason': 'EOD Exit'})

    return trades

def main():
    symbols = get_symbols_from_gsheet(GOOGLE_SHEET_ID)

    print("Backtesting using Exit Case 1 (RSI < 70 + Close < BB Upper):")
    all_trades_case_1 = []
    for symbol in symbols:
        print(f"Processing {symbol}...")
        df = load_data(symbol)
        if df is None or len(df) < 20:
            print(f"Not enough data for {symbol}, skipping.")
            continue
        df = compute_indicators(df)
        trades = backtest_symbol(df, symbol, exit_case_1)
        all_trades_case_1.extend(trades)

    total_pnl_1 = sum(t['pnl'] for t in all_trades_case_1)
    wins_1 = sum(1 for t in all_trades_case_1 if t['pnl'] > 0)
    win_rate_1 = (wins_1 / len(all_trades_case_1) * 100) if all_trades_case_1 else 0
    print(f"\nExit Case 1 Summary: Total Trades={len(all_trades_case_1)}, Total PnL={total_pnl_1:.2f}, Win Rate={win_rate_1:.2f}%")

    print("\nBacktesting using Exit Case 2 (Case 1 + Chandelier Exit):")
    all_trades_case_2 = []
    for symbol in symbols:
        print(f"Processing {symbol}...")
        df = load_data(symbol)
        if df is None or len(df) < 20:
            print(f"Not enough data for {symbol}, skipping.")
            continue
        df = compute_indicators(df)
        trades = backtest_symbol(df, symbol, exit_case_2)
        all_trades_case_2.extend(trades)

    total_pnl_2 = sum(t['pnl'] for t in all_trades_case_2)
    wins_2 = sum(1 for t in all_trades_case_2 if t['pnl'] > 0)
    win_rate_2 = (wins_2 / len(all_trades_case_2) * 100) if all_trades_case_2 else 0
    print(f"\nExit Case 2 Summary: Total Trades={len(all_trades_case_2)}, Total PnL={total_pnl_2:.2f}, Win Rate={win_rate_2:.2f}%")

if __name__ == "__main__":
    main()
