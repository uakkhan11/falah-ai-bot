import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
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
SLIPPAGE = 0.0005  # 0.05% slippage per trade side
MAX_POSITIONS = 5
MAX_TRADES = 2000

# Refined SL_ATR_MULT range near best result for fine tuning
SL_ATR_MULT_RANGE = np.arange(1.8, 2.3, 0.1)
CHANDLER_MULT = 3.0  # Fixed chandelier multiplier

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
    return df[df['date'] >= cutoff]

def compute_indicators(df, chandelier_mult=CHANDLER_MULT):
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
    return (row['macd_line'] > 0 and
            row['macd_signal'] > 0 and
            40 <= row['rsi_14'] <= 70 and
            row['high'] > row['high_1d_ago'] and
            row['low'] > row['low_1d_ago'] and
            row['volume'] > row['volume_1d_ago'] and
            row['close'] >= row['bb_lower'])

def momentum_exit_filter(row):
    # Momentum exit: RSI drops below 70 and close below upper bollinger band
    return (row['rsi_14'] < 70) and (row['close'] < row['bb_upper'])

def exit_logic(row, sl_atr_mult, initial_sl_dict, pos):
    # Determine current dynamic stop loss price:
    entry_price = pos['entry_price']
    atr = pos['atr']
    # Calculate ATR-based stop loss
    atr_sl_price = entry_price - sl_atr_mult * atr
    # Get initial 2% fixed stop loss price
    initial_sl_price = initial_sl_dict[pos['entry_date']]
    # Chandelier exit
    chandelier_sl = row['chandelier_exit']
    # Effective stop loss price is max of initial SL, ATR SL, chandelier (furthest downside stop)
    effective_sl = max(initial_sl_price, atr_sl_price, chandelier_sl)
    # Check stop loss hit
    stop_loss_hit = (row['low'] <= effective_sl)
    # Check momentum exit
    momentum_exit = momentum_exit_filter(row)
    return stop_loss_hit, momentum_exit, effective_sl

def backtest_symbol(df, symbol, sl_atr_mult):
    cash = INITIAL_CAPITAL
    positions = []
    trades = []
    trade_count = 0
    entry_signal_count = 0
    exit_signal_count = 0
    equity_curve = []
    portfolio_value = cash

    # Track initial stop loss for each position entry date (fixed 2% below entry)
    initial_sl_dict = {}

    for i in range(1, len(df)):
        row = df.iloc[i]
        date = row['date']
        price = row['close']

        positions_to_close = []
        for pos in positions:
            stop_loss_hit, momentum_exit, effective_sl = exit_logic(row, sl_atr_mult, initial_sl_dict, pos)
            if stop_loss_hit or momentum_exit:
                exit_price = effective_sl if stop_loss_hit else price
                # Add slippage cost: assume slippage on both buy and sell
                pnl = pos['direction'] * (exit_price - pos['entry_price']) * pos['shares']
                pnl -= abs(pnl) * (TRANSACTION_COST + SLIPPAGE) * 2
                cash += exit_price * pos['shares'] * (1 - TRANSACTION_COST - SLIPPAGE)
                trades.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'pnl': pnl,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'direction': 'Long' if pos['direction'] == 1 else 'Short',
                    'exit_reason': 'Stop Loss' if stop_loss_hit else 'Momentum Exit'
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
                cash -= shares * price * (1 + TRANSACTION_COST + SLIPPAGE)
                pos = {'entry_date': date, 'entry_price': price,
                       'shares': shares, 'direction': 1, 'atr': atr}
                positions.append(pos)
                entry_signal_count += 1
                # Set initial 2% fixed stop loss price for this entry
                initial_sl_dict[date] = price * 0.98
        if trade_count >= MAX_TRADES:
            break

        # Update portfolio value (equity curve)
        position_value = sum((row['close'] - pos['entry_price']) * pos['shares'] * pos['direction'] for pos in positions)
        portfolio_value = cash + position_value
        equity_curve.append(portfolio_value)

    # Close all remaining positions at last close
    final_date = df.iloc[-1]['date']
    final_close = df.iloc[-1]['close']
    for pos in positions:
        pnl = pos['direction'] * (final_close - pos['entry_price']) * pos['shares']
        pnl -= abs(pnl) * (TRANSACTION_COST + SLIPPAGE) * 2
        cash += final_close * pos['shares'] * (1 - TRANSACTION_COST - SLIPPAGE)
        trades.append({
            'symbol': symbol,
            'entry_date': pos['entry_date'],
            'exit_date': final_date,
            'pnl': pnl,
            'entry_price': pos['entry_price'],
            'exit_price': final_close,
            'direction': 'Long' if pos['direction'] == 1 else 'Short',
            'exit_reason': 'EOD Exit'
        })
        equity_curve.append(cash)

    # Calculate performance metrics
    pnl_list = [t['pnl'] for t in trades]
    total_pnl = sum(pnl_list)
    wins = len([p for p in pnl_list if p > 0])
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades else 0

    # Calculate drawdown
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (running_max - equity_array) / running_max
    max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0

    # Calculate average trade duration
    trade_durations = []
    for t in trades:
        duration = (t['exit_date'] - t['entry_date']).days
        if duration >= 0:
            trade_durations.append(duration)
    avg_duration = np.mean(trade_durations) if trade_durations else 0

    print(f"\n{symbol} - SL_ATR_MULT={sl_atr_mult}:")
    print(f"Trades: {total_trades}, Entries: {entry_signal_count}, Exits (excl stop loss): {exit_signal_count}")
    print(f"Total PnL: {total_pnl:.2f}, Win Rate: {win_rate:.2f}%, Max Drawdown: {max_drawdown:.2f}%, Avg Trade Duration: {avg_duration:.2f} days")

    return total_pnl, win_rate, max_drawdown, avg_duration

def parameter_sweep():
    symbols = get_symbols_from_gsheet(GOOGLE_SHEET_ID)
    results = []

    for sl_atr_mult in SL_ATR_MULT_RANGE:
        total_pnl_sum = 0
        total_drawdown_list = []
        total_duration_list = []
        for symbol in symbols:
            df = load_data(symbol)
            if df is None or len(df) < 20:
                continue
            df = compute_indicators(df)
            pnl, win_rate, max_dd, avg_dur = backtest_symbol(df, symbol, sl_atr_mult)
            total_pnl_sum += pnl
            total_drawdown_list.append(max_dd)
            total_duration_list.append(avg_dur)
        avg_drawdown = np.mean(total_drawdown_list) if total_drawdown_list else 0
        avg_duration = np.mean(total_duration_list) if total_duration_list else 0
        results.append({
            'SL_ATR_MULT': sl_atr_mult,
            'TOTAL_PnL': total_pnl_sum,
            'AVG_MAX_DD_pct': avg_drawdown,
            'AVG_TRADE_DURATION_days': avg_duration
        })

    results = sorted(results, key=lambda x: x['TOTAL_PnL'], reverse=True)

    print("\n=== Refined Parameter Sweep Results ===")
    print("SL_ATR_MULT | TOTAL_PnL | AVG_MAX_DD% | AVG_TRADE_DURATION_days")
    for r in results:
        print(f"{r['SL_ATR_MULT']:11.2f} | {r['TOTAL_PnL']:9.2f} | {r['AVG_MAX_DD_pct']:10.2f} | {r['AVG_TRADE_DURATION_days']:21.2f}")

if __name__ == "__main__":
    parameter_sweep()
