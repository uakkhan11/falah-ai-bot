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
SL_ATR_MULT = 2.8  # ATR stop loss multiplier
INITIAL_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.01 * INITIAL_CAPITAL  # 1% risk per trade
TRANSACTION_COST = 0.001  # 0.1% per transaction
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
    df['bb_lower'] = ta.bbands(df['close'], length=20, std=2)['BBL_20_2.0']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Shifted columns for price & volume comparison
    df['high_1d_ago'] = df['high'].shift(1)
    df['low_1d_ago'] = df['low'].shift(1)
    df['high_2d_ago'] = df['high'].shift(2)
    df['low_2d_ago'] = df['low'].shift(2)
    df['volume_1d_ago'] = df['volume'].shift(1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def bullish_entry_filter(row):
    return (
        row['macd_line'] > 0 and
        row['macd_signal'] > 0 and
        row['rsi_14'] > 55 and
        row['high'] > row['high_1d_ago'] and
        row['low'] > row['low_1d_ago'] and
        row['high'] > row['high_2d_ago'] and
        row['low'] > row['low_2d_ago'] and
        row['volume'] > row['volume_1d_ago'] and
        row['rsi_14'] < 70 and
        row['close'] >= row['bb_lower']
    )

def backtest_symbol(df, symbol):
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    trade_count = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        date = row['date']
        price = row['close']

        # Close positions if stop loss hit
        for pos in positions[:]:
            direction = pos['direction']
            entry_price = pos['entry_price']
            atr = pos['atr']
            stop_loss_price = entry_price - direction * SL_ATR_MULT * atr

            if (direction == 1 and df.iloc[i]['low'] <= stop_loss_price) or \
               (direction == -1 and df.iloc[i]['high'] >= stop_loss_price):
                exit_price = stop_loss_price
                pnl = direction * (exit_price - entry_price) * pos['shares']
                pnl -= abs(pnl) * TRANSACTION_COST * 2  # buy + sell cost
                cash += exit_price * pos['shares'] * (1 - TRANSACTION_COST)
                trades.append({'symbol': symbol,
                               'entry_date': pos['entry_date'],
                               'exit_date': date,
                               'pnl': pnl,
                               'entry_price': entry_price,
                               'exit_price': exit_price,
                               'direction': 'Long' if direction == 1 else 'Short',
                               'exit_reason': 'Stop Loss'})
                positions.remove(pos)
                trade_count += 1
                if trade_count >= MAX_TRADES:
                    break
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
    all_trades = []

    for symbol in symbols:
        print(f"Processing {symbol}...")
        df = load_data(symbol)
        if df is None or len(df) < 20:
            print(f"Not enough data for {symbol}, skipping.")
            continue
        df = compute_indicators(df)
        trades = backtest_symbol(df, symbol)
        all_trades.extend(trades)

        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = (wins / len(trades) * 100) if trades else 0
        print(f"{symbol}: Trades={len(trades)}, Total PnL={total_pnl:.2f}, Win Rate={win_rate:.2f}%")

    overall_pnl = sum(t['pnl'] for t in all_trades)
    overall_wins = sum(1 for t in all_trades if t['pnl'] > 0)
    overall_trades = len(all_trades)
    overall_win_rate = (overall_wins / overall_trades * 100) if overall_trades else 0

    print("\n--- Overall Backtest Summary ---")
    print(f"Total trades: {overall_trades}")
    print(f"Total PnL: {overall_pnl:.2f}")
    print(f"Overall Win Rate: {overall_win_rate:.2f}%")

if __name__ == "__main__":
    main()
