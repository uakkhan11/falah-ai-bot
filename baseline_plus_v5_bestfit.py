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

# Volatility threshold for regime switching
VOLATILITY_THRESHOLD = 1.0  # Adjust as needed based on ATR scale

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

# Individual exit conditions
def chandelier_exit(row):
    return row['close'] < row['chandelier_exit']

def rsi_bb_exit(row):
    return (row['rsi_14'] < 70) and (row['close'] < row['bb_upper'])

def supertrend_exit(row):
    return row['supertrend_direction'] < 0

# Hybrid exit combines all exits
def hybrid_exit_logic(row):
    return chandelier_exit(row) or rsi_bb_exit(row) or supertrend_exit(row)

# Regime-adaptive exit switches based on ATR level (volatility)
def regime_adaptive_exit(row):
    if row['atr'] is None or np.isnan(row['atr']):
        return False
    if row['atr'] > VOLATILITY_THRESHOLD:
        # Higher volatility: use chandelier exit
        return chandelier_exit(row)
    else:
        # Lower volatility: use RSI + BB exit
        return rsi_bb_exit(row)

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
                pos = {'entry_date': date,
                       'entry_price': price,
                       'shares': shares,
                       'direction': 1,
                       'atr': atr}
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
    summary_data = {
        'Hybrid Exit': {'trades': [], 'entries': 0, 'exits': 0},
        'Regime-Adaptive Exit': {'trades': [], 'entries': 0, 'exits': 0}
    }

    print("=== Running Hybrid Exit Backtest ===")
    for symbol in symbols:
        df = load_data(symbol)
        if df is None or len(df) < 20:
            continue
        df = compute_indicators(df)
        trades, entries, exits = backtest_symbol(df, symbol, hybrid_exit_logic, "Hybrid Exit")
        summary_data['Hybrid Exit']['trades'].extend(trades)
        summary_data['Hybrid Exit']['entries'] += entries
        summary_data['Hybrid Exit']['exits'] += exits

    print("\n=== Running Regime-Adaptive Exit Backtest ===")
    for symbol in symbols:
        df = load_data(symbol)
        if df is None or len(df) < 20:
            continue
        df = compute_indicators(df)
        trades, entries, exits = backtest_symbol(df, symbol, regime_adaptive_exit, "Regime-Adaptive Exit")
        summary_data['Regime-Adaptive Exit']['trades'].extend(trades)
        summary_data['Regime-Adaptive Exit']['entries'] += entries
        summary_data['Regime-Adaptive Exit']['exits'] += exits

    print("\n\n=== SUMMARY OF ALL ADAPTIVE EXIT CASES ===")
    for case_label, data in summary_data.items():
        total_trades = len(data['trades'])
        total_entries = data['entries']
        total_exits = data['exits']
        total_pnl = sum(t['pnl'] for t in data['trades'])
        wins = sum(1 for t in data['trades'] if t['pnl'] > 0)
        win_rate = (wins / total_trades * 100) if total_trades else 0
        print(f"\n{case_label} Overall:")
        print(f"Total entries signaled: {total_entries}")
        print(f"Total exit logic triggered (excl. stop loss): {total_exits}")
        print(f"Total trades executed: {total_trades}")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Win rate: {win_rate:.2f}%")

if __name__ == "__main__":
    main()
