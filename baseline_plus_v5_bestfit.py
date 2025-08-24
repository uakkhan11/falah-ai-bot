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
INITIAL_CAPITAL = 200_000
RISK_PER_TRADE = 0.01 * INITIAL_CAPITAL
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
MAX_POSITIONS = 5
MAX_TRADES = 2000

# Regime volatility threshold for stop loss sensitivity
VOLATILITY_THRESHOLD = 1.0

def get_symbols_from_gsheet(sheet_id, worksheet_name="HalalList"):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
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
    df['supertrend_direction'] = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)['SUPERTd_10_3.0']
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def bullish_entry_filter(row):
    return (
        row['macd_line'] > 0 and
        row['macd_signal'] > 0 and
        40 <= row['rsi_14'] <= 70 and
        row['close'] >= row['bb_lower']
    )

def momentum_trend_exit(row):
    # Exit if momentum fading or trend reversing
    rsi_exit = (row['rsi_14'] < 70) and (row['close'] < row['bb_upper'])
    trend_exit = row['supertrend_direction'] < 0
    return rsi_exit or trend_exit

def adaptive_sl_atr_multiplier(atr):
    # Tighter stop loss in calm markets, looser in volatile markets
    if atr > VOLATILITY_THRESHOLD:
        return 3.0
    else:
        return 2.0

def backtest_symbol(df, symbol):
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
            sl_atr_mult = adaptive_sl_atr_multiplier(pos['atr'])

            entry_price = pos['entry_price']
            stop_loss_price = entry_price - sl_atr_mult * pos['atr']
            stop_loss_hit = df.iloc[i]['low'] <= stop_loss_price

            exit_signal = momentum_trend_exit(row)
            if stop_loss_hit or exit_signal:
                exit_price = stop_loss_price if stop_loss_hit else price
                pnl = (exit_price - entry_price) * pos['shares']
                pnl -= abs(pnl) * (TRANSACTION_COST + SLIPPAGE) * 2
                cash += exit_price * pos['shares'] * (1 - TRANSACTION_COST - SLIPPAGE)
                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': 'Stop Loss' if stop_loss_hit else 'Momentum/Trend Exit'
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
                position_size = RISK_PER_TRADE / (adaptive_sl_atr_multiplier(row['atr']) * row['atr'])
                shares = min(cash / price, position_size)
                if shares < 1:
                    continue
                cash -= shares * price * (1 + TRANSACTION_COST + SLIPPAGE)
                positions.append({'entry_date': date, 'entry_price': price, 'shares': shares, 'atr': row['atr']})
                entry_signal_count += 1

        position_value = sum((row['close'] - pos['entry_price']) * pos['shares'] for pos in positions)
        portfolio_value = cash + position_value
        equity_curve.append(portfolio_value)

    final_date = df.iloc[-1]['date']
    final_close = df.iloc[-1]['close']
    for pos in positions:
        pnl = (final_close - pos['entry_price']) * pos['shares']
        pnl -= abs(pnl) * (TRANSACTION_COST + SLIPPAGE) * 2
        cash += final_close * pos['shares'] * (1 - TRANSACTION_COST - SLIPPAGE)
        trades.append({
            'symbol': symbol,
            'entry_date': pos['entry_date'],
            'exit_date': final_date,
            'pnl': pnl,
            'entry_price': pos['entry_price'],
            'exit_price': final_close,
            'exit_reason': 'EOD Exit'
        })
        equity_curve.append(cash)

    total_pnl = sum(t['pnl'] for t in trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades else 0
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (running_max - equity_array) / running_max
    max_drawdown = np.max(drawdowns) * 100 if drawdowns.size else 0
    durations = [(t['exit_date'] - t['entry_date']).days for t in trades if (t['exit_date'] - t['entry_date']).days >= 0]
    avg_duration = np.mean(durations) if durations else 0

    print(f"\n{symbol} Backtest Summary:")
    print(f"Total Trades: {total_trades}, Win Rate: {win_rate:.2f}%")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Avg Trade Duration: {avg_duration:.2f} days")

    return total_pnl, win_rate, max_drawdown, avg_duration

def main():
    symbols = get_symbols_from_gsheet(GOOGLE_SHEET_ID)
    total_pnl = 0
    total_trades = 0
    total_wins = 0
    total_drawdowns = []
    total_durations = []

    for symbol in symbols:
        df = load_data(symbol)
        if df is None or len(df) < 20:
            continue
        df = compute_indicators(df)
        pnl, win_rate, max_dd, avg_dur = backtest_symbol(df, symbol)
        total_pnl += pnl
        total_drawdowns.append(max_dd)
        total_durations.append(avg_dur)

    avg_drawdown = np.mean(total_drawdowns) if total_drawdowns else 0
    avg_duration = np.mean(total_durations) if total_durations else 0
    print("\n=== Aggregate Strategy Performance ===")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Average Max Drawdown: {avg_drawdown:.2f}%")
    print(f"Average Trade Duration: {avg_duration:.2f} days")

if __name__ == "__main__":
    main()
