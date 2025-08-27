import os
import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings
import json

warnings.filterwarnings("ignore")

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

def get_symbols_from_daily_data():
    daily_files = os.listdir(DATA_PATHS['daily'])
    return [os.path.splitext(f)[0] for f in daily_files if f.endswith('.csv')]

def compute_indicators(df):
    df = df.copy()
    df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill()
    df['high'] = pd.to_numeric(df['high'], errors='coerce').ffill()
    df['low'] = pd.to_numeric(df['low'], errors='coerce').ffill()
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    close = df['close'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    volume = df['volume'].values.astype(np.float64)
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
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = upperband, middleband, lowerband
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['obv'] = talib.OBV(close, volume)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def load_and_filter_data(symbol, years=5):
    cutoff_date = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365 * years)
    daily_df = pd.read_csv(os.path.join(DATA_PATHS['daily'], f"{symbol}.csv"), parse_dates=['date'])
    hourly_df = pd.read_csv(os.path.join(DATA_PATHS['1hour'], f"{symbol}.csv"), parse_dates=['date'])
    m15_df = pd.read_csv(os.path.join(DATA_PATHS['15minute'], f"{symbol}.csv"), parse_dates=['date'])
    for df in [daily_df, hourly_df, m15_df]:
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= cutoff_date].reset_index(drop=True)
    return daily_df, hourly_df, m15_df

def prepare_data_for_symbol(symbol, years=5):
    daily_df, hourly_df, m15_df = load_and_filter_data(symbol, years)
    daily_df = compute_indicators(daily_df)
    hourly_df = compute_indicators(hourly_df)
    m15_df = compute_indicators(m15_df)
    daily_df.dropna(subset=['ema8','ema20'], inplace=True)
    hourly_df.dropna(subset=['ema8','ema20'], inplace=True)
    m15_df.dropna(subset=['ema8','ema20'], inplace=True)
    return daily_df, hourly_df, m15_df

def get_worst_market_periods(df, percentile=5):
    df['return'] = df['close'].pct_change()
    worst_days = df[df['return'] < df['return'].quantile(percentile / 100.0)]
    return worst_days[['date', 'return']]

class BacktestTrailingStrategy:
    def __init__(self, df, initial_capital=100000, trailing_pct=0.01):
        self.df = df.sort_values('date').reset_index(drop=True)
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.entry_price = None
        self.highest_price = None
        self.trades = []
        self.trailing_pct = trailing_pct

    def run(self):
        for i in range(1, len(self.df)):
            row = self.df.iloc[i]
            if self.position == 0:
                if row.get('ema8', None) is not None and row.get('ema20', None) is not None:
                    if row['ema8'] > row['ema20'] and self.df.iloc[i-1]['ema8'] <= self.df.iloc[i-1]['ema20']:
                        qty = int(self.cash / row['close'])
                        if qty > 0:
                            self.position = qty
                            self.entry_price = row['close']
                            self.highest_price = row['close']
                            self.cash -= qty * row['close']
                            self.trades.append({'type': 'BUY', 'date': row['date'], 'price': row['close'], 'qty': qty})
            else:
                self.highest_price = max(self.highest_price, row['close'])
                trailing_stop = self.highest_price * (1 - self.trailing_pct)
                if row['close'] <= trailing_stop:
                    pnl = (row['close'] - self.entry_price) * self.position
                    self.cash += self.position * row['close']
                    self.trades.append({'type': 'SELL', 'date': row['date'], 'price': row['close'], 'qty': self.position, 'pnl': pnl, 'exit_reason': 'Trailing Stop'})
                    self.position = 0
                    self.entry_price = None
        if self.position > 0:
            pnl = (self.df.iloc[-1]['close'] - self.entry_price) * self.position
            self.cash += self.position * self.df.iloc[-1]['close']
            self.trades.append({'type': 'SELL', 'date': self.df.iloc[-1]['date'], 'price': self.df.iloc[-1]['close'], 'qty': self.position, 'pnl': pnl, 'exit_reason': "EOD Exit"})
        return self.trades

def walk_forward_test_symbol(symbol, window_days=180, test_days=30, trailing_pct=0.01):
    daily_df, hourly_df, m15_df = prepare_data_for_symbol(symbol)
    m15_df = m15_df.sort_values('date').reset_index(drop=True)
    results = []
    start_idx = 0
    while start_idx + window_days + test_days < len(m15_df):
        train_df = m15_df.iloc[start_idx : start_idx + window_days]
        test_df = m15_df.iloc[start_idx + window_days : start_idx + window_days + test_days]
        bt_train = BacktestTrailingStrategy(train_df, trailing_pct=trailing_pct)
        trades_train = bt_train.run()
        bt_test = BacktestTrailingStrategy(test_df, trailing_pct=trailing_pct)
        trades_test = bt_test.run()
        results.append({'window_start': train_df['date'].iloc[0], 'window_end': test_df['date'].iloc[-1],
                        'train_PnL': sum([t.get('pnl',0) for t in trades_train if 'pnl' in t]),
                        'test_PnL': sum([t.get('pnl',0) for t in trades_test if 'pnl' in t]),
                        'total_trades_test': len([t for t in trades_test if t['type']=='SELL'])})
        start_idx += test_days
    return results

def symbol_allocation_optimization(symbol_stats, total_capital=100000):
    total_pf = sum(max(stats.get('Profit Factor', 0.01), 0.01) for stats in symbol_stats.values())
    allocs = {}
    for symbol, stats in symbol_stats.items():
        pf = max(stats.get('Profit Factor', 0.01), 0.01)
        allocs[symbol] = pf / total_pf * total_capital
    return allocs

def write_final_report(summary, stats, allocations, walkforward, worstdays, filename="final_report.txt"):
    with open(filename, "w") as f:
        f.write("="*80 + "\nFINAL STRATEGY REPORT\n" + "="*80 + "\n")
        f.write("Capital Used: {}\n".format(summary.get("capital_used", 100000)))
        f.write("Total Profit: {}\n\n".format(summary.get('total_profit',0)))
        f.write("Symbol-Wise Allocations:\n")
        for s,a in allocations.items():
            f.write("  {}: {:.2f}\n".format(s, a))
        f.write("\nWalk-Forward Test Results:\n")
        for sym,res_list in walkforward.items():
            f.write(f"{sym}:\n")
            for res in res_list:
                f.write(f"  {res}\n")
        f.write("\nWorst-Day Market Results (drawdown periods):\n")
        for sym, wd_results in worstdays.items():
            f.write(f"  {sym}: {wd_results}\n")
        f.write("\nPer Symbol Trade Stats:\n")
        for sym, stat in stats.items():
            f.write(f"{sym}:\n")
            for k,v in stat.items():
                f.write(f"  {k}: {v}\n")
        f.write("\nRecommendations:\n")
        f.write("1. Favor symbols resilient in worst periods for live trading.\n")
        f.write("2. Use trailing stop for better risk control.\n")
        f.write("3. Allocate capital dynamically based on live profit factor.\n")
        f.write("4. Paper trade before going live.\n")
        f.write("5. Continue feature engineering to boost ML performance.\n")
    print(f"\nFinal report saved: {filename}")

if __name__ == "__main__":
    CAPITAL = 100000
    symbols = get_symbols_from_daily_data()
    summary = {"capital_used": CAPITAL, "total_profit": 0}
    symbol_stats = {}
    walkforward_results = {}
    worstday_results = {}

    for symbol in symbols:
        daily_df, hourly_df, m15_df = prepare_data_for_symbol(symbol)
        bt = BacktestTrailingStrategy(m15_df, initial_capital=CAPITAL, trailing_pct=0.01)
        trades = bt.run()
        total_pnl = sum(t.get('pnl', 0) for t in trades if 'pnl' in t)
        wins = sum(t.get("pnl",0)>0 for t in trades if t.get('type',"")=="SELL")
        losses = sum(t.get("pnl",0)<=0 for t in trades if t.get('type',"")=="SELL")
        profit_factor = sum(t.get('pnl',0) for t in trades if t.get('pnl',0)>0) / (-sum(t.get('pnl',0) for t in trades if t.get('pnl',0)<=0)+1e-6)
        symbol_stats[symbol] = {
            "Total Trades": len([t for t in trades if t.get('type',"")=="SELL"]),
            "Winning Trades": wins,
            "Losing Trades": losses,
            "Win Rate %": 100.0*wins/(wins+losses+1e-6),
            "Profit Factor": profit_factor,
            "Total PnL": total_pnl
        }
        summary['total_profit'] += total_pnl

        wf_results = walk_forward_test_symbol(symbol, window_days=180, test_days=30, trailing_pct=0.01)
        walkforward_results[symbol] = wf_results

        worst_days = get_worst_market_periods(daily_df)
        if not worst_days.empty:
            wd_df = m15_df[m15_df['date'].isin(worst_days['date'])]
            trades_worst = BacktestTrailingStrategy(wd_df, trailing_pct=0.01).run()
            worst_pnl = sum(t.get('pnl',0) for t in trades_worst if 'pnl' in t)
            worstday_results[symbol] = {"trades": len(trades_worst), "PnL": worst_pnl}
        else:
            worstday_results[symbol] = {}

    allocations = symbol_allocation_optimization(symbol_stats, total_capital=CAPITAL)
    write_final_report(summary, symbol_stats, allocations, walkforward_results, worstday_results, filename="final_report.txt")
