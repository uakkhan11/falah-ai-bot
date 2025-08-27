import os
import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from collections import Counter

import warnings
import json

warnings.filterwarnings("ignore")

BASE_DIR = "/root/falah-ai-bot"
DATA_PATHS = {
    'daily': os.path.join(BASE_DIR, "swing_data"),
    '1hour': os.path.join(BASE_DIR, "intraday_swing_data"),
    '15minute': os.path.join(BASE_DIR, "scalping_data"),
}

CAPITAL = 100000

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
    def __init__(self, df, initial_capital=CAPITAL, trailing_pct=0.01):
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

def run_fixed_stop_backtest(df, initial_capital=CAPITAL, stop_loss_pct=0.01, profit_target_pct=0.015):
    cash = initial_capital
    position = 0
    entry_price = None
    trades = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        if position == 0:
            if row.get('ema8') is not None and row.get('ema20') is not None:
                prev_row = df.iloc[i-1]
                if prev_row['ema8'] <= prev_row['ema20'] and row['ema8'] > row['ema20']:
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
        last_price = df.iloc[-1]['close']
        pnl = (last_price - entry_price) * position
        cash += position * last_price
        trades.append({'type': 'SELL', 'date': df.iloc[-1]['date'], 'price': last_price, 'qty': position, 'pnl': pnl, 'exit_reason': 'EOD Exit'})
    return trades, cash

def calc_trade_stats(trades):
    df = pd.DataFrame(trades)
    if df.empty or 'pnl' not in df.columns:
        return {}
    df_closed = df[df['type'] == 'SELL']
    wins = (df_closed['pnl'] > 0).sum()
    losses = (df_closed['pnl'] <= 0).sum()
    profit_factor = df_closed[df_closed['pnl'] > 0]['pnl'].sum() / (-df_closed[df_closed['pnl'] <= 0]['pnl'].sum() + 1e-9)
    stats = {
        'Total Trades': len(df_closed),
        'Winning Trades': wins,
        'Losing Trades': losses,
        'Win Rate %': round(100 * wins / len(df_closed), 2) if len(df_closed) > 0 else 0,
        'Profit Factor': round(profit_factor, 2),
        'Avg PnL per Trade': round(df_closed['pnl'].mean(), 2),
        'Total PnL': round(df_closed['pnl'].sum(), 2),
    }
    return stats

def create_ml_features(m15_df, hourly_df):
    hourly_df = hourly_df.set_index('date')
    m15_df = m15_df.set_index('date')
    for col in ['ema8', 'ema20', 'rsi_14', 'adx']:
        if col in hourly_df.columns:
            m15_df['hour_' + col] = hourly_df[col].reindex(m15_df.index, method='ffill')
    m15_df['future_return'] = m15_df['close'].shift(-10) / m15_df['close'] - 1
    m15_df['label'] = (m15_df['future_return'] > 0.01).astype(int)
    m15_df.dropna(inplace=True)
    labels = m15_df['label']
    features = m15_df.drop(columns=['label', 'future_return', 'open', 'high', 'low', 'volume'], errors='ignore')
    return features, labels

def ml_train_and_filter(df, hourly_df, threshold=0.7):
    X, y = create_ml_features(df, hourly_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    proba = model.predict_proba(X_test)[:, 1]
    filtered_indices = proba > threshold
    filtered_y_test = y_test[filtered_indices]
    filtered_acc = accuracy_score(filtered_y_test, y_pred[filtered_indices]) if len(filtered_y_test) > 0 else 0
    filtered_prec = precision_score(filtered_y_test, y_pred[filtered_indices]) if len(filtered_y_test) > 0 else 0
    filtered_rec = recall_score(filtered_y_test, y_pred[filtered_indices]) if len(filtered_y_test) > 0 else 0
    return {
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'filtered_accuracy': filtered_acc,
        'filtered_precision': filtered_prec,
        'filtered_recall': filtered_rec
    }

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

def symbol_allocation_optimization(symbol_stats, total_capital=CAPITAL):
    total_pf = sum(max(stats.get('Profit Factor', 0.01), 0.01) for stats in symbol_stats.values())
    allocs = {}
    for symbol, stats in symbol_stats.items():
        pf = max(stats.get('Profit Factor', 0.01), 0.01)
        allocs[symbol] = pf / total_pf * total_capital
    return allocs

def write_final_report(summary, stats, allocations, walkforward, worstdays, comparison_reports, filename="final_report.txt"):
    with open(filename, "w") as f:
        f.write("="*80 + "\nFINAL STRATEGY REPORT\n" + "="*80 + "\n")
        f.write(f"Capital Used: {summary.get('capital_used', CAPITAL)}\n")
        f.write(f"Total Profit: {summary.get('total_profit',0)}\n\n")
        f.write("Symbol-Wise Allocations:\n")
        for s, a in allocations.items():
            f.write(f"  {s}: {a:.2f}\n")
        f.write("\nWalk-Forward Test Results:\n")
        for sym, res_list in walkforward.items():
            f.write(f"{sym}:\n")
            for res in res_list:
                f.write(f"  {res}\n")
        f.write("\nWorst-Day Market Results:\n")
        for sym, wd_res in worstdays.items():
            f.write(f"  {sym}: {wd_res}\n")
        f.write("\nPer Symbol Trade Stats:\n")
        for sym, stat in stats.items():
            f.write(f"{sym}:\n")
            for k, v in stat.items():
                f.write(f"  {k}: {v}\n")
        f.write("\nFixed vs Trailing Stop Loss Comparison:\n")
        for sym, comp in comparison_reports.items():
            f.write(f"{sym}:\n")
            for k, v in comp.items():
                f.write(f"  {k}: {v}\n")
        f.write("\nRecommendations:\n")
        f.write("1. Favor symbols resilient in worst market periods.\n")
        f.write("2. Use trailing stop loss to manage risk better.\n")
        f.write("3. Dynamic capital allocation based on recent profitability.\n")
        f.write("4. Apply ML filtering to improve trade precision.\n")
        f.write("5. Extensive paper testing before live deployment.\n")
    print(f"Report saved to {filename}")

def compare_fixed_vs_trailing(df, hourly_df, capital=CAPITAL, stop_loss_pct=0.01, profit_target_pct=0.015, trailing_pct=0.01, ml_threshold=0.7):
    fixed_trades, fixed_cash = run_fixed_stop_backtest(df, capital, stop_loss_pct, profit_target_pct)
    fixed_stats = calc_trade_stats(fixed_trades)
    bt = BacktestTrailingStrategy(df, initial_capital=capital, trailing_pct=trailing_pct)
    trailing_trades = bt.run()
    trailing_stats = calc_trade_stats(trailing_trades)
    ml_metrics = ml_train_and_filter(df, hourly_df, ml_threshold)
    return {
        'Fixed SL Stats': fixed_stats,
        'Trailing SL Stats': trailing_stats,
        'ML Performance': ml_metrics,
        'Capital Fixed SL': fixed_cash,
        'Capital Trailing SL': bt.cash
    }

if __name__ == "__main__":
    symbols = get_symbols_from_daily_data()
    summary = {'capital_used': CAPITAL, 'total_profit': 0}
    symbol_stats = {}
    walkforward_results = {}
    worstday_results = {}
    comparison_reports = {}

    for symbol in symbols:
        print(f"Processing symbol: {symbol}")
        daily_df, hourly_df, m15_df = prepare_data_for_symbol(symbol)
        bt = BacktestTrailingStrategy(m15_df, initial_capital=CAPITAL, trailing_pct=0.01)
        trades = bt.run()
        total_pnl = sum(t.get('pnl', 0) for t in trades if 'pnl' in t)
        wins = sum(t.get('pnl', 0) > 0 for t in trades if t.get('type', '') == "SELL")
        losses = sum(t.get('pnl', 0) <= 0 for t in trades if t.get('type', '') == "SELL")
        profit_factor = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0) / (-sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) <= 0) + 1e-6)
        symbol_stats[symbol] = {
            'Total Trades': len([t for t in trades if t.get('type', '') == "SELL"]),
            'Winning Trades': wins,
            'Losing Trades': losses,
            'Win Rate %': 100.0 * wins / (wins + losses + 1e-6),
            'Profit Factor': profit_factor,
            'Total PnL': total_pnl
        }
        summary['total_profit'] += total_pnl
        walkforward_results[symbol] = walk_forward_test_symbol(symbol, window_days=180, test_days=30, trailing_pct=0.01)
        worst_days = get_worst_market_periods(daily_df)
        if not worst_days.empty:
            wd_df = m15_df[m15_df['date'].isin(worst_days['date'])]
            trades_worst = BacktestTrailingStrategy(wd_df, trailing_pct=0.01).run()
            worst_pnl = sum(t.get('pnl', 0) for t in trades_worst if 'pnl' in t)
            worstday_results[symbol] = {'Trades': len(trades_worst), 'PnL': worst_pnl}
        else:
            worstday_results[symbol] = {}
        comparison_reports[symbol] = compare_fixed_vs_trailing(m15_df, hourly_df, capital=CAPITAL)
    allocations = symbol_allocation_optimization(symbol_stats, total_capital=CAPITAL)
    write_final_report(summary, symbol_stats, allocations, walkforward_results, worstday_results, comparison_reports, filename="full_detailed_report.txt")


def generate_consolidated_summary(symbol_stats, comparison_reports, filename="consolidated_summary.txt"):
    total_symbols = len(symbol_stats)
    total_trades = 0
    total_wins = 0
    total_losses = 0
    profit_factors = []
    total_pnl = 0.0

    # Accumulators for ML metrics and features
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    feature_counter = Counter()

    # Aggregate numeric stats
    for symbol, stats in symbol_stats.items():
        total_trades += stats.get('Total Trades', 0)
        total_wins += stats.get('Winning Trades', 0)
        total_losses += stats.get('Losing Trades', 0)
        pf = stats.get('Profit Factor', None)
        if pf is not None:
            profit_factors.append(pf)
        total_pnl += stats.get('Total PnL', 0.0)

        # Add ML metrics and features from comparison reports
        comp = comparison_reports.get(symbol, {})
        ml_perf = comp.get('ML Performance', {})
        if ml_perf:
            all_accuracies.append(ml_perf.get('accuracy', 0))
            all_precisions.append(ml_perf.get('precision', 0))
            all_recalls.append(ml_perf.get('recall', 0))
            # Count feature importance keys, if any
            # Assuming ml_perf includes something like 'feature_importance' dict if extended
            # If not available, consider extending your ML function to provide it
            # Example: feature_counter.update(ml_perf.get('feature_importance', {}))

    avg_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    avg_profit_factor = sum(profit_factors)/len(profit_factors) if profit_factors else 0
    avg_accuracy = sum(all_accuracies)/len(all_accuracies) if all_accuracies else 0
    avg_precision = sum(all_precisions)/len(all_precisions) if all_precisions else 0
    avg_recall = sum(all_recalls)/len(all_recalls) if all_recalls else 0

    with open(filename, "w") as f:
        f.write("="*80 + "\n")
        f.write("🚀 CONSOLIDATED SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total Symbols Analyzed: {total_symbols}\n")
        f.write(f"Total Trades Executed: {total_trades}\n")
        f.write(f"Aggregate Wins: {total_wins} | Losses: {total_losses} | Win Rate: {avg_win_rate:.2f}%\n")
        f.write(f"Average Profit Factor: {avg_profit_factor:.2f}\n")
        f.write(f"Total Portfolio PnL: {total_pnl:.2f}\n\n")

        f.write("ML Performance (averaged over symbols):\n")
        f.write(f"    Accuracy: {avg_accuracy:.4f}\n")
        f.write(f"    Precision: {avg_precision:.4f}\n")
        f.write(f"    Recall: {avg_recall:.4f}\n\n")

        if feature_counter:
            f.write("Top ML Features (aggregated importance):\n")
            for feat, count in feature_counter.most_common(10):
                f.write(f"    {feat}: {count}\n")
            f.write("\n")

        # Optionally add more aggregate info like best symbol, best PF, etc.
        best_symbol = max(symbol_stats.items(), key=lambda x: x[1].get('Total PnL', float('-inf')))[0] if symbol_stats else None
        if best_symbol:
            f.write(f"Top Performing Symbol by PnL: {best_symbol} with PnL: {symbol_stats[best_symbol].get('Total PnL',0):.2f}\n")

        f.write("\nRecommendations:\n")
        f.write(" - Focus on symbols with strong PnL and consistent win rate.\n")
        f.write(" - Use ML filtering and trailing stops to enhance risk control.\n")
        f.write(" - Dynamic position sizing improves portfolio robustness.\n")
        f.write(" - Continue enhancing features and validating in walk-forward tests.\n")

    print(f"Consolidated summary saved to {filename}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_detailed_backtest_report(
    trades, price_series, initial_capital, 
    ml_metrics=None, ml_feature_importance=None, 
    filename="backtest_report.txt"
):
    df = pd.DataFrame(trades)
    df_closed = df[df['type'] == 'SELL']

    # 1. Summary Overview
    total_trades = len(df_closed)
    wins = (df_closed['pnl'] > 0).sum()
    losses = total_trades - wins
    win_pct = 100 * wins / total_trades if total_trades > 0 else 0
    net_profit = df_closed['pnl'].sum()
    roi = (net_profit / initial_capital) * 100

    # Equity for all dates, assuming starting capital
    equity = (price_series / price_series.iloc[0]) * initial_capital
    daily_returns = equity.pct_change().dropna()
    days = (price_series.index[-1] - price_series.index[0]).days
    trading_days_per_year = 252

    annualized_return = (equity.iloc[-1] / equity.iloc[0]) ** (trading_days_per_year / len(daily_returns)) - 1

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    sharpe_ratio = np.sqrt(trading_days_per_year) * daily_returns.mean() / daily_returns.std()
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.sqrt(trading_days_per_year) * daily_returns.mean() / downside_returns.std() if not downside_returns.empty else np.nan

    gross_profit = df_closed[df_closed['pnl'] > 0]['pnl'].sum()
    gross_loss = -df_closed[df_closed['pnl'] <= 0]['pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_duration = df_closed['duration'].mean() if 'duration' in df_closed else np.nan

    # Consecutive wins/losses
    def max_consecutive_wins_losses(series):
        max_wins = max_losses = current_streak = 0
        prev_win = None
        for val in series:
            if val > 0:
                if prev_win == True:
                    current_streak += 1
                else:
                    current_streak = 1
                max_wins = max(max_wins, current_streak)
                prev_win = True
            else:
                if prev_win == False:
                    current_streak += 1
                else:
                    current_streak = 1
                max_losses = max(max_losses, current_streak)
                prev_win = False
        return max_wins, max_losses

    max_wins, max_losses = max_consecutive_wins_losses(df_closed['pnl'])

    avg_win = df_closed[df_closed['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = df_closed[df_closed['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0

    # 2. Equity Curve Visualization (save plot)
    try:
        plt.figure(figsize=(10,6))
        plt.plot(equity.index, equity.values, label='Equity Curve')
        buy_dates = df[df['type']=='BUY']['date']
        sell_dates = df[df['type']=='SELL']['date']
        plt.scatter(buy_dates, equity.reindex(buy_dates, method='nearest'), marker='^', color='g', label='Buy')
        plt.scatter(sell_dates, equity.reindex(sell_dates, method='nearest'), marker='v', color='r', label='Sell')
        plt.title("Equity Curve with Trade Markers")
        plt.legend()
        plt.grid(True)
        plot_filename = filename.replace('.txt','.png')
        plt.savefig(plot_filename)
        plt.close()
    except Exception as e:
        plot_filename = None

    # Write report text
    with open(filename, 'w') as f:
        f.write("=== Backtest Detailed Report ===\n\n")
        f.write("1. Summary Overview\n")
        f.write(f"Strategy Name: Custom Strategy\n")
        f.write(f"Backtest Period: {price_series.index[0].date()} - {price_series.index[-1].date()}\n")
        f.write(f"Initial Capital: {initial_capital}\n")
        f.write(f"Total Trades Executed: {total_trades}\n")
        f.write(f"Winning Trades (%): {win_pct:.2f}%\n")
        f.write(f"Losing Trades (%): {100 - win_pct:.2f}%\n")
        f.write(f"Net Profit/Loss: {net_profit:.2f}\n")
        f.write(f"Return on Investment (ROI %): {roi:.2f}%\n")
        f.write(f"Annualized Return: {annualized_return*100:.2f}%\n")
        f.write(f"Max Drawdown (%): {max_drawdown:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
        f.write(f"Sortino Ratio: {sortino_ratio:.4f}\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n")
        f.write(f"Average Trade Duration (days): {avg_duration:.2f}\n\n")

        f.write(f"Equity curve plot saved as: {plot_filename}\n\n")

        f.write("3. Performance Metrics\n")
        f.write(f"Total Net Profit: {net_profit:.2f}\n")
        f.write(f"Annualized Return (%): {annualized_return*100:.2f}\n")
        f.write(f"Max Drawdown (%): {max_drawdown:.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
        f.write(f"Sortino Ratio: {sortino_ratio:.4f}\n")
        f.write(f"Win Rate (%): {win_pct:.2f}\n")
        f.write(f"Average Win (%): {avg_win:.2f}\n")
        f.write(f"Average Loss (%): {avg_loss:.2f}\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n")
        f.write(f"Max Consecutive Wins: {max_wins}\n")
        f.write(f"Max Consecutive Losses: {max_losses}\n")
        f.write(f"Average Trade Duration: {avg_duration:.2f}\n\n")

        f.write("4. Trade Log\n")
        f.write("Trade # | Entry Date | Exit Date | Entry Price | Exit Price | P&L (%) | Duration (days) | Notes\n")
        for idx, trade in enumerate(df_closed.itertuples(), start=1):
            entry = df[(df['date'] < trade.date) & (df['type']=='BUY')].iloc[-1] if not df[(df['date'] < trade.date) & (df['type']=='BUY')].empty else None
            notes = getattr(trade, 'exit_reason', '')
            f.write(f"{idx} | {entry['date'].date() if entry is not None else 'N/A'} | {trade.date.date()} | {entry['price'] if entry is not None else 'N/A'} | {trade.price} | {trade.pnl:.2f} | {trade.duration if hasattr(trade, 'duration') else 'N/A'} | {notes}\n")

        if ml_metrics is not None:
            f.write("\n5. Indicator / ML Performance Details\n")
            for metric_name, val in ml_metrics.items():
                if metric_name != 'model':
                    f.write(f"{metric_name}: {val:.4f}\n")

        if ml_feature_importance is not None:
            f.write("Feature Importance:\n")
            for feat, imp in ml_feature_importance.items():
                f.write(f"  {feat}: {imp}\n")

        # Sections 6-8 would require more inputs and can be added similarly

        f.write("\n=== End of Report ===\n")

    # Print report to terminal (optional)
    with open(filename, 'r') as f:
        print(f.read())

