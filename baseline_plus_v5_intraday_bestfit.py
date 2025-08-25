import pandas as pd
import numpy as np
from datetime import datetime
import os

BASE_DIR = "/root/falah-ai-bot"  # Update this path to your base directory

DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
}

def get_symbols_from_daily_data():
    # List all CSV files in the daily data folder and get filenames without extension
    daily_files = os.listdir(DATA_PATHS['daily'])
    symbols = [os.path.splitext(f)[0] for f in daily_files if f.endswith('.csv')]
    return symbols

def add_indicators(df):
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['vol_sma20'] = df['volume'].rolling(window=20).mean()
    return df

def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

class BacktestStrategy:
    def __init__(self, daily_df, hourly_df, df_15m, initial_capital=100000):
        self.daily_df = daily_df.sort_values('date').reset_index(drop=True)
        self.hourly_df = add_indicators(hourly_df.sort_values('date').reset_index(drop=True))
        self.df_15m = add_indicators(df_15m.sort_values('date').reset_index(drop=True))
        self.initial_capital = initial_capital

        self.cash = initial_capital
        self.position = 0
        self.entry_price = 0
        self.entry_date = None
        self.trades = []
        self.exit_reasons = {'StopLoss': 0, 'SignalExit': 0, 'EOD Exit': 0}

    def run_backtest(self):
        daily_row = self.daily_df.iloc[-1]
        if daily_row['close'] <= daily_row['close'].ewm(span=200, adjust=False).mean():
            return []  # Daily trend down, skip symbol

        last_hourly = self.hourly_df.iloc[-1]
        if not (last_hourly['ema8'] > last_hourly['ema20'] and last_hourly['rsi'] > 50 and last_hourly['volume'] > last_hourly['vol_sma20']):
            return []  # Hourly confirmation failed

        df = self.df_15m

        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]

            # Entry signal: EMA8 crosses above EMA20 15m
            entry_signal = prev['ema8'] <= prev['ema20'] and curr['ema8'] > curr['ema20']

            if self.position == 0 and entry_signal:
                qty = int(self.cash / curr['close'])
                if qty > 0:
                    self.entry_price = curr['close']
                    self.position = qty
                    self.entry_date = curr['date']
                    self.cash -= qty * self.entry_price
                    self.trades.append({'type': 'BUY', 'date': curr['date'], 'price': self.entry_price, 'qty': qty})
            elif self.position > 0:
                current_price = curr['close']
                profit_target = self.entry_price * 1.01
                stop_loss = self.entry_price * 0.9975

                if current_price >= profit_target:
                    pnl = (current_price - self.entry_price) * self.position
                    self.exit_reasons['SignalExit'] += 1
                    self._exit_trade(current_price, pnl, curr['date'], "Profit Target")
                elif current_price <= stop_loss:
                    pnl = (current_price - self.entry_price) * self.position
                    self.exit_reasons['StopLoss'] += 1
                    self._exit_trade(current_price, pnl, curr['date'], "Stop Loss")

        # Force exit end of day (last candle)
        if self.position > 0:
            current_price = df.iloc[-1]['close']
            pnl = (current_price - self.entry_price) * self.position
            self.exit_reasons['EOD Exit'] += 1
            self._exit_trade(current_price, pnl, df.iloc[-1]['date'], "EOD Exit")

        return self.trades

    def _exit_trade(self, price, pnl, date, reason):
        self.cash += self.position * price
        trade_duration = (pd.to_datetime(date) - pd.to_datetime(self.entry_date)).days + 1
        self.trades.append({
            'type': 'SELL', 'date': date, 'price': price, 'qty': self.position, 'pnl': pnl, 'duration': trade_duration, 'exit_reason': reason
        })
        self.position = 0
        self.entry_price = 0
        self.entry_date = None

def overall_performance(trades, initial_capital):
    if not trades:
        print("No trades executed")
        return

    df = pd.DataFrame(trades)
    total_trades = len(df) // 2  # Buy + Sell pairs
    wins = df[df['pnl'] > 0]['pnl'].count()
    losses = df[df['pnl'] <= 0]['pnl'].count()
    win_rate = wins / total_trades * 100 if total_trades else 0
    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = -df[df['pnl'] <= 0]['pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    expectancy = df['pnl'].mean()
    best_trade = df['pnl'].max()
    worst_trade = df['pnl'].min()
    avg_duration = df[df['pnl'].notnull()]['duration'].mean()

    total_return = (df['pnl'].sum() / initial_capital) * 100

    print("\n= OVERALL BACKTEST SUMMARY =")
    print(f"Total Trades Taken     : {total_trades}")
    print(f"Winning Trades         : {wins}")
    print(f"Losing Trades          : {losses}")
    print(f"Win Rate               : {win_rate:.2f}%")
    print(f"Profit Factor          : {profit_factor:.2f}")
    print(f"Expectancy (per trade) : {expectancy:.2f}")
    print(f"Best Trade (PnL)       : {best_trade:.2f}")
    print(f"Worst Trade (PnL)      : {worst_trade:.2f}")
    print(f"Avg Trade Duration     : {avg_duration:.2f} days")

    print("\n---- Equity Summary ----")
    print(f"Total Return           : {total_return:.2f}%")
    # Placeholder for CAGR, Max Drawdown, Sharpe Ratio (requires time series data)
    print(f"CAGR                   : N/A")
    print(f"Max Drawdown           : N/A")
    print(f"Sharpe Ratio           : N/A")
    print("=====")

if __name__ == "__main__":
    initial_capital = 100000
    symbols = get_symbols_from_daily_data()  # Load symbols by daily data CSV filenames

    all_trades = []

    for symbol in symbols:
        try:
            daily_csv = os.path.join(DATA_PATHS['daily'], f"{symbol}.csv")
            hourly_csv = os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv")
            scalping_csv = os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv")

            daily_df = pd.read_csv(daily_csv, parse_dates=['date'])
            hourly_df = pd.read_csv(hourly_csv, parse_dates=['date'])
            scalping_df = pd.read_csv(scalping_csv, parse_dates=['date'])

            strategy = BacktestStrategy(daily_df, hourly_df, scalping_df, initial_capital)
            trades = strategy.run_backtest()
            all_trades.extend(trades)

        except Exception as e:
            print(f"Skipping {symbol} due to error: {e}")

    overall_performance(all_trades, initial_capital)
