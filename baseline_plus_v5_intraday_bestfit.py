import os
import pandas as pd
import numpy as np
import talib
from datetime import datetime

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

def get_symbols_from_daily_data():
    daily_files = os.listdir(DATA_PATHS['daily'])
    symbols = [os.path.splitext(f)[0] for f in daily_files if f.endswith('.csv')]
    return symbols

def compute_indicators(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    df['ema8'] = talib.EMA(close, timeperiod=8)
    df['ema20'] = talib.EMA(close, timeperiod=20)
    df['rsi_14'] = talib.RSI(close, timeperiod=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    df['roc'] = talib.ROC(close, timeperiod=10)
    df['cmo'] = talib.CMO(close, timeperiod=14)
    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20)
    df['bb_upper'] = upperband
    df['bb_middle'] = middleband
    df['bb_lower'] = lowerband
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['obv'] = talib.OBV(close, volume)

    return df

def load_and_filter_data(symbol, years=5):
    cutoff_date = pd.Timestamp.utcnow() - pd.Timedelta(days=365*years)
    daily_df = pd.read_csv(os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"), parse_dates=['date'])
    hourly_df = pd.read_csv(os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"), parse_dates=['date'])
    m15_df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"), parse_dates=['date'])

    # Ensure datetime dtype
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    hourly_df['date'] = pd.to_datetime(hourly_df['date'])
    m15_df['date'] = pd.to_datetime(m15_df['date'])

    daily_df = daily_df[daily_df['date'] >= cutoff_date].reset_index(drop=True)
    hourly_df = hourly_df[hourly_df['date'] >= cutoff_date].reset_index(drop=True)
    m15_df = m15_df[m15_df['date'] >= cutoff_date].reset_index(drop=True)

    return daily_df, hourly_df, m15_df

class BacktestStrategy:
    def __init__(self, daily_df, hourly_df, m15_df, initial_capital=100000):
        self.daily_df = compute_indicators(daily_df.sort_values('date').reset_index(drop=True))
        self.hourly_df = compute_indicators(hourly_df.sort_values('date').reset_index(drop=True))
        self.df_15m = compute_indicators(m15_df.sort_values('date').reset_index(drop=True))
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.entry_price = 0
        self.entry_date = None
        self.trades = []
        self.exit_reasons = {'StopLoss': 0, 'ProfitTarget': 0, 'EOD Exit': 0}

    def run_backtest(self):
        # Calculate daily EMA200 for trend filter
        daily_ema200 = self.daily_df['close'].ewm(span=200, adjust=False).mean()
        daily_close_last = self.daily_df['close'].iloc[-1]
        if daily_close_last <= daily_ema200.iloc[-1]:
            return []  # Skip - daily trend not up

        last_hourly = self.hourly_df.iloc[-1]
        if not (last_hourly['ema8'] > last_hourly['ema20'] 
                and last_hourly['rsi_14'] > 50 
                and last_hourly['volume'] > last_hourly['volume'].rolling(window=20).mean().iloc[-1]):
            return []  # Skip - hourly confirmation not met

        df = self.df_15m
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]

            # Long entry signal: EMA8 crosses above EMA20 on 15m
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
                profit_target = self.entry_price * 1.01   # 1% profit target
                stop_loss = self.entry_price * 0.9975     # 0.25% stop loss
                
                if current_price >= profit_target:
                    pnl = (current_price - self.entry_price) * self.position
                    self.exit_reasons['ProfitTarget'] += 1
                    self._exit_trade(current_price, pnl, curr['date'], "Profit Target")

                elif current_price <= stop_loss:
                    pnl = (current_price - self.entry_price) * self.position
                    self.exit_reasons['StopLoss'] += 1
                    self._exit_trade(current_price, pnl, curr['date'], "Stop Loss")

        # Exit any open position at last candle close
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
            'type': 'SELL', 'date': date, 'price': price, 'qty': self.position,
            'pnl': pnl, 'duration': trade_duration, 'exit_reason': reason
        })
        self.position = 0
        self.entry_price = 0
        self.entry_date = None

def generate_report_symbol(trades, symbol, initial_capital):
    df = pd.DataFrame(trades)
    if df.empty:
        return None
    total_trades = len(df) // 2  # buy+sell pairs
    wins = df[df['pnl'] > 0]['pnl'].count()
    losses = df[df['pnl'] <= 0]['pnl'].count()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = -df[df['pnl'] <= 0]['pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    expectancy = df['pnl'].mean()
    best_trade = df['pnl'].max()
    worst_trade = df['pnl'].min()
    avg_duration = df['duration'].mean()
    total_return = (df['pnl'].sum() / initial_capital) * 100
    report = {
        'Symbol': symbol,
        'Total Trades': total_trades,
        'Wins': wins,
        'Losses': losses,
        'Win Rate %': win_rate,
        'Profit Factor': profit_factor,
        'Expectancy': expectancy,
        'Best Trade PnL': best_trade,
        'Worst Trade PnL': worst_trade,
        'Avg Trade Duration (days)': avg_duration,
        'Total Return %': total_return,
    }
    return report

def overall_report(all_trades, initial_capital):
    all_df = pd.DataFrame(all_trades)
    if all_df.empty:
        print("No trades executed.")
        return
    summary = generate_report_symbol(all_trades, "ALL_SYMBOLS", initial_capital)
    print("\n===== OVERALL BACKTEST REPORT =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    all_df.to_csv("all_symbols_trades.csv", index=False)
    print("\nDetailed trade data saved to 'all_symbols_trades.csv' for further analysis.")

if __name__ == "__main__":
    initial_capital = 100000
    symbols = get_symbols_from_daily_data()
    all_trades = []
    reports = []
    for symbol in symbols:
        try:
            daily_df, hourly_df, m15_df = load_and_filter_data(symbol, years=5)
            if len(daily_df) < 100 or len(hourly_df) < 100 or len(m15_df) < 100:
                print(f"Skipping {symbol} due to insufficient data.")
                continue
            strategy = BacktestStrategy(daily_df, hourly_df, m15_df, initial_capital)
            trades = strategy.run_backtest()
            all_trades.extend(trades)
            report = generate_report_symbol(trades, symbol, initial_capital)
            if report:
                reports.append(report)
                print(f"Report for {symbol}: {report}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    overall_report(all_trades, initial_capital)
