import os
import pandas as pd
import numpy as np
import talib
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}
CAPITAL = 100000

def get_symbols_from_local_data():
    daily_files = os.listdir(DATA_PATHS['daily'])
    symbols = [os.path.splitext(f)[0] for f in daily_files if f.endswith('.csv')]
    return symbols

def load_and_prepare_symbol(symbol, years=5):
    path = os.path.join(DATA_PATHS['daily'], f"{symbol}.csv")
    df = pd.read_csv(path, parse_dates=['date'])
    # Make cutoff timezone naive to match df['date'] dtype and avoid comparison issues
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365 * years)
    df = df[df['date'] >= cutoff].sort_values('date').reset_index(drop=True)
    # Compute indicators
    close = df['close'].astype(float).values
    df['ema8'] = talib.EMA(close, timeperiod=8)
    df['ema20'] = talib.EMA(close, timeperiod=20)
    df.dropna(subset=['ema8', 'ema20'], inplace=True)
    return df.reset_index(drop=True)
    
def backtest_trailing_stop(df, initial_capital, trailing_pct=0.01):
    cash = initial_capital
    position = 0
    entry_price = None
    highest_price = None
    trades = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        if position == 0:
            if row['ema8'] > row['ema20'] and df.iloc[i-1]['ema8'] <= df.iloc[i-1]['ema20']:
                qty = int(cash / row['close'])
                if qty > 0:
                    position = qty
                    entry_price = row['close']
                    highest_price = row['close']
                    cash -= qty * row['close']
                    trades.append({'type': 'BUY', 'date': row['date'], 'price': row['close'], 'qty': qty})
        else:
            highest_price = max(highest_price, row['close'])
            trailing_stop = highest_price * (1 - trailing_pct)
            if row['close'] <= trailing_stop:
                pnl = (row['close'] - entry_price) * position
                cash += position * row['close']
                trades.append({'type': 'SELL', 'date': row['date'], 'price': row['close'], 'qty': position, 'pnl': pnl, 'exit_reason': 'Trailing Stop'})
                position = 0
                entry_price = None
    if position > 0:
        pnl = (df.iloc[-1]['close'] - entry_price) * position
        cash += position * df.iloc[-1]['close']
        trades.append({'type': 'SELL', 'date': df.iloc[-1]['date'], 'price': df.iloc[-1]['close'], 'qty': position, 'pnl': pnl, 'exit_reason': 'EOD Exit'})
    return trades

def backtest_fixed_stop(df, initial_capital, stop_loss_pct=0.01, profit_target_pct=0.015):
    cash = initial_capital
    position = 0
    entry_price = None
    trades = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        if position == 0:
            if row['ema8'] > row['ema20'] and df.iloc[i-1]['ema8'] <= df.iloc[i-1]['ema20']:
                qty = int(cash / row['close'])
                if qty > 0:
                    position = qty
                    entry_price = row['close']
                    cash -= qty * row['close']
                    trades.append({'type': 'BUY', 'date': row['date'], 'price': row['close'], 'qty': qty})
        else:
            stop_loss = entry_price * (1 - stop_loss_pct)
            profit_target = entry_price * (1 + profit_target_pct)
            current_price = row['close']
            if current_price <= stop_loss:
                pnl = (current_price - entry_price) * position
                cash += position * current_price
                trades.append({'type': 'SELL', 'date': row['date'], 'price': current_price, 'qty': position, 'pnl': pnl, 'exit_reason': 'Stop Loss'})
                position = 0
                entry_price = None
            elif current_price >= profit_target:
                pnl = (current_price - entry_price) * position
                cash += position * current_price
                trades.append({'type': 'SELL', 'date': row['date'], 'price': current_price, 'qty': position, 'pnl': pnl, 'exit_reason': 'Profit Target'})
                position = 0
                entry_price = None
    if position > 0:
        pnl = (df.iloc[-1]['close'] - entry_price) * position
        cash += position * df.iloc[-1]['close']
        trades.append({'type': 'SELL', 'date': df.iloc[-1]['date'], 'price': df.iloc[-1]['close'], 'qty': position, 'pnl': pnl, 'exit_reason': 'EOD Exit'})
    return trades

def calc_trade_stats(trades):
    df = pd.DataFrame(trades)
    if df.empty or 'pnl' not in df.columns:
        return {}
    df_closed = df[df['type']=='SELL']
    wins = (df_closed['pnl'] > 0).sum()
    losses = (df_closed['pnl'] <= 0).sum()
    profit_factor = df_closed[df_closed['pnl'] > 0]['pnl'].sum() / (-df_closed[df_closed['pnl'] <= 0]['pnl'].sum() + 1e-9)
    stats = {
        'Total Trades': len(df_closed),
        'Winning Trades': wins,
        'Losing Trades': losses,
        'Win Rate %': round(100 * wins / len(df_closed), 2),
        'Profit Factor': round(profit_factor, 2),
        'Avg PnL per Trade': round(df_closed['pnl'].mean(), 2),
        'Total PnL': round(df_closed['pnl'].sum(), 2)
    }
    return stats

def write_strategy_report(strategy_name, trades, stats, filename):
    with open(filename, "w") as f:
        f.write(f"Strategy: {strategy_name}\n")
        f.write(f"Total Trades: {stats.get('Total Trades', 0)}\n")
        f.write(f"Winning Trades: {stats.get('Winning Trades', 0)}\n")
        f.write(f"Losing Trades: {stats.get('Losing Trades', 0)}\n")
        f.write(f"Win Rate (%): {stats.get('Win Rate %', 0)}\n")
        f.write(f"Profit Factor: {stats.get('Profit Factor', 0)}\n")
        f.write(f"Average PnL per Trade: {stats.get('Avg PnL per Trade', 0)}\n")
        f.write(f"Total PnL: {stats.get('Total PnL', 0)}\n\n")
        f.write("Trade Log:\n")
        f.write("Entry Date | Exit Date | Entry Price | Exit Price | Qty | PnL | Exit Reason\n")
        df = pd.DataFrame(trades)
        for idx, row in df.iterrows():
            if row['type'] == 'SELL':
                buy_rows = df[(df['type']=='BUY') & (df['date'] < row['date'])]
                if not buy_rows.empty:
                    buy = buy_rows.iloc[-1]
                    f.write(f"{buy['date'].date()} | {row['date'].date()} | {buy['price']} | {row['price']} | {row['qty']} | {row.get('pnl',0):.2f} | {row.get('exit_reason','')}\n")

def load_all_symbols_data(symbols, years=5):
    all_data = {}
    for sym in symbols:
        df = load_and_prepare_symbol(sym, years)
        if df is not None and not df.empty:
            all_data[sym] = df
    return all_data

def run_portfolio_backtest(all_data, backtest_func, initial_capital):
    total_trades = []
    total_cash = initial_capital * len(all_data)  # approximate sum capital across symbols
    for sym, df in all_data.items():
        trades = backtest_func(df, initial_capital)
        total_trades.extend(trades)
    return total_trades, total_cash

def aggregate_trade_stats(trades):
    df = pd.DataFrame(trades)
    if df.empty:
        return {}
    df_sells = df[df['type'] == 'SELL']
    wins = (df_sells['pnl'] > 0).sum()
    losses = (df_sells['pnl'] <= 0).sum()
    total_trades = len(df_sells)
    win_rate = 100 * wins / total_trades if total_trades > 0 else 0
    total_pnl = df_sells['pnl'].sum()
    return {
        'Total Trades': total_trades,
        'Winning Trades': wins,
        'Losing Trades': losses,
        'Win Rate %': round(win_rate, 2),
        'Total PnL': round(total_pnl, 2)
    }

def write_final_comparison_report(all_stats, filename="final_comparison_report.txt"):
    with open(filename, "w") as f:
        f.write("Symbol | Strategy | Total Trades | Win Rate % | Avg PnL per Trade | Total PnL\n")
        f.write("--------------------------------------------------------------------------\n")
        for sym, stats in all_stats.items():
            for strat, s in stats.items():
                f.write(f"{sym} | {strat} | {s.get('Total Trades','N/A')} | {s.get('Win Rate %','N/A')} | {s.get('Avg PnL per Trade','N/A')} | {s.get('Total PnL','N/A')}\n")

if __name__ == "__main__":
    symbols = get_symbols_from_local_data()
    all_data = load_all_symbols_data(symbols)
    for symbol in symbols:
        print(f"Processing symbol: {symbol}")
        df = load_and_prepare_symbol(symbol)
        
        # Trailing Stop Backtest
        trades_trailing = backtest_trailing_stop(df, CAPITAL)
        stats_trailing = calc_trade_stats(trades_trailing)
        write_strategy_report(f"Trailing Stop - {symbol}", trades_trailing, stats_trailing, f"report_trailing_{symbol}.txt")
        
        # Fixed Stop Backtest
        trades_fixed = backtest_fixed_stop(df, CAPITAL)
        stats_fixed = calc_trade_stats(trades_fixed)
        write_strategy_report(f"Fixed Stop - {symbol}", trades_fixed, stats_fixed, f"report_fixed_{symbol}.txt")
        
        combined_stats[symbol] = {
            "Trailing Stop": stats_trailing,
            "Fixed Stop": stats_fixed
        }

        print("Portfolio Level Summary")
        print("------------------------")
        print("Trailing Stop Strategy:")
        for k,v in trailing_stats.items():
            print(f"{k}: {v}")
        print("\nFixed Stop Strategy:")
        for k,v in fixed_stats.items():
            print(f"{k}: {v}")
    write_final_comparison_report(combined_stats)
    print("All reports generated.")
