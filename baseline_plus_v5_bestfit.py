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

# ATR + Chandelier Settings
ATR_LENGTH = 14
CHAND_LENGTH = 22
ATR_MULT = 2.0   # starting stop loss multiple


# ============================================================
# Google Sheet Symbols
# ============================================================

def get_symbols_from_gsheet(sheet_id, worksheet_name="HalalList"):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDS_JSON, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(worksheet_name)
    symbols = worksheet.col_values(1)
    return [s.strip() for s in symbols if s.strip()]


# ============================================================
# Data & Indicators
# ============================================================

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
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_LENGTH)

    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_line'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']

    df['rsi_14'] = ta.rsi(df['close'], length=14)

    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_lower'] = bbands['BBL_20_2.0']
    df['bb_upper'] = bbands['BBU_20_2.0']

    st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3.0)
    df['supertrend_direction'] = st['SUPERTd_10_3.0']

    highest_high = df['high'].rolling(CHAND_LENGTH).max()
    df['chandelier_exit'] = highest_high - df['atr'] * ATR_MULT

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# Entry & Exit Filters
# ============================================================

def bullish_entry_filter(row):
    return (
        row['macd_line'] > 0 and
        row['macd_signal'] > 0 and
        40 <= row['rsi_14'] <= 70 and
        row['close'] >= row['bb_lower']
    )

def exit_signal(row):
    rsi_exit = (row['rsi_14'] < 70) and (row['close'] < row['bb_upper'])
    trend_exit = row['supertrend_direction'] < 0
    return rsi_exit or trend_exit


# ============================================================
# True Portfolio-Level Backtester
# ============================================================

def backtest_portfolio(symbols):
    # Load all data once
    data = {}
    for sym in symbols:
        df = load_data(sym)
        if df is None or len(df) < 50:
            continue
        data[sym] = compute_indicators(df)

    # Get common timeline (intersection of trading days)
    all_dates = sorted(set().union(*[set(df['date']) for df in data.values()]))
    if not all_dates:
        print("No data available for backtest")
        return

    # Portfolio state
    cash = INITIAL_CAPITAL
    positions = {}   # {symbol: {entry_date, entry_price, shares}}
    trades = []
    equity_curve = []
    trade_count = 0

    # Walk day by day across all symbols
    for date in all_dates:
        # Filter each symbol for current day row
        daily_rows = {}
        for sym, df in data.items():
            row = df[df['date'] == date]
            if not row.empty:
                daily_rows[sym] = row.iloc[0]

        # === Check exits first ===
        positions_to_remove = []
        for sym, pos in positions.items():
            if sym not in daily_rows:  # no data for today
                continue
            row = daily_rows[sym]
            entry_price = pos['entry_price']
            shares = pos['shares']

            chand_sl = row['chandelier_exit']
            atr_sl = entry_price - ATR_MULT * row['atr']
            dynamic_sl = max(chand_sl, atr_sl)

            stop_loss_hit = row['low'] <= dynamic_sl
            forced_exit = exit_signal(row)

            if stop_loss_hit or forced_exit:
                exit_price = dynamic_sl if stop_loss_hit else row['close']
                pnl = (exit_price - entry_price) * shares

                pnl -= abs(entry_price * shares) * (TRANSACTION_COST + SLIPPAGE)
                pnl -= abs(exit_price * shares) * (TRANSACTION_COST + SLIPPAGE)
                cash += exit_price * shares * (1 - TRANSACTION_COST - SLIPPAGE)

                trades.append({
                    'symbol': sym,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'exit_reason': 'StopLoss' if stop_loss_hit else 'SignalExit'
                })
                positions_to_remove.append(sym)
                trade_count += 1
                if trade_count >= MAX_TRADES:
                    break

        for sym in positions_to_remove:
            del positions[sym]

        if trade_count >= MAX_TRADES:
            break

        # === New entries ===
        for sym, row in daily_rows.items():
            if len(positions) >= MAX_POSITIONS:
                break
            if sym in positions:
                continue
            if bullish_entry_filter(row) and cash > 0:
                risk_per_share = ATR_MULT * row['atr']
                position_size = RISK_PER_TRADE / risk_per_share
                shares = int(min(cash // row['close'], position_size))
                if shares >= 1:
                    cost = shares * row['close'] * (1 + TRANSACTION_COST + SLIPPAGE)
                    cash -= cost
                    positions[sym] = {
                        'entry_date': date,
                        'entry_price': row['close'],
                        'shares': shares
                    }

        # === Update equity ===
        pos_value = 0
        for sym, pos in positions.items():
            if sym in daily_rows:
                row = daily_rows[sym]
                pos_value += (row['close'] - pos['entry_price']) * pos['shares'] + pos['shares'] * pos['entry_price']
        equity_curve.append(cash + pos_value)

    # === Liquidate end-of-period ===
    if positions:
        final_date = all_dates[-1]
        for sym, pos in positions.items():
            df = data[sym]
            final_row = df[df['date'] == final_date]
            if final_row.empty:
                continue
            final_row = final_row.iloc[0]
            final_close = final_row['close']
            pnl = (final_close - pos['entry_price']) * pos['shares']
            pnl -= abs(pos['entry_price'] * pos['shares']) * (TRANSACTION_COST + SLIPPAGE)
            pnl -= abs(final_close * pos['shares']) * (TRANSACTION_COST + SLIPPAGE)
            cash += final_close * pos['shares'] * (1 - TRANSACTION_COST - SLIPPAGE)
            trades.append({
                'symbol': sym,
                'entry_date': pos['entry_date'],
                'exit_date': final_date,
                'pnl': pnl,
                'entry_price': pos['entry_price'],
                'exit_price': final_close,
                'exit_reason': 'EOD Exit'
            })
        equity_curve.append(cash)

    return trades, equity_curve


# ============================================================
# Reporting
# ============================================================

def overall_report(all_trades, equity_curve):
    df = pd.DataFrame(all_trades)
    if df.empty:
        print("No trades executed!")
        return

    total_trades = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]

    win_rate = 100 * len(wins) / total_trades if total_trades else 0
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) else np.inf
    expectancy = df['pnl'].mean()
    best_trade = df['pnl'].max()
    worst_trade = df['pnl'].min()

    durations = (df['exit_date'] - df['entry_date']).dt.days
    avg_duration = durations.mean() if len(durations) else 0
    exit_counts = df['exit_reason'].value_counts().to_dict()

    equity_array = np.array(equity_curve)
    total_return = (equity_array[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    running_max = np.maximum.accumulate(equity_array)
    dd = (running_max - equity_array) / running_max
    max_dd = dd.max() * 100 if dd.size else 0

    n_years = len(equity_array) / 252
    CAGR = ((equity_array[-1] / INITIAL_CAPITAL) ** (1/n_years) - 1) * 100 if n_years > 0 else 0

    daily_returns = pd.Series(equity_array).pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if not daily_returns.empty else 0

    print("\n=== OVERALL PORTFOLIO BACKTEST SUMMARY ===")
    print(f"Total Trades Taken     : {total_trades}")
    print(f"Winning Trades         : {len(wins)}")
    print(f"Losing Trades          : {len(losses)}")
    print(f"Win Rate               : {win_rate:.2f}%")
    print(f"Profit Factor          : {profit_factor:.2f}")
    print(f"Expectancy (per trade) : {expectancy:.2f}")
    print(f"Best Trade (PnL)       : {best_trade:.2f}")
    print(f"Worst Trade (PnL)      : {worst_trade:.2f}")
    print(f"Avg Trade Duration     : {avg_duration:.2f} days")
    print(f"Exit Reasons           : {exit_counts}")
    print("---- Equity Summary ----")
    print(f"Total Return           : {total_return:.2f}%")
    print(f"CAGR                   : {CAGR:.2f}%")
    print(f"Max Drawdown           : {max_dd:.2f}%")
    print(f"Sharpe Ratio           : {sharpe:.2f}")
    print("===========================================\n")


# ============================================================
# Main
# ============================================================

def main():
    symbols = get_symbols_from_gsheet(GOOGLE_SHEET_ID)
    trades, equity_curve = backtest_portfolio(symbols)
    overall_report(trades, equity_curve)


if __name__ == "__main__":
    main()
