import os
import pandas as pd
import numpy as np
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import Counter
import warnings
import matplotlib.pyplot as plt
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

def add_trade_durations(trades):
    # trades is a list of dicts with 'type', 'date', 'price', etc.
    enhanced_trades = []
    open_trades = []

    for t in trades:
        t = t.copy()
        # Convert date to pandas Timestamp if not already
        if not isinstance(t['date'], pd.Timestamp):
            t['date'] = pd.to_datetime(t['date'])
        if t['type'] == 'BUY':
            open_trades.append(t)
        elif t['type'] == 'SELL' and open_trades:
            # Match this sell to the last open buy
            entry = open_trades.pop(0)
            duration_days = (t['date'] - entry['date']).days
            t['duration'] = duration_days
            # Append both entry and exit trades with duration info
            # Add duration also to entry for reporting
            entry['duration'] = duration_days
            enhanced_trades.append(entry)
            enhanced_trades.append(t)
        else:
            # SELL without matching BUY, no duration
            t['duration'] = None
            enhanced_trades.append(t)
    # Append any unmatched open trades as is (no duration)
    for t in open_trades:
        t['duration'] = None
        enhanced_trades.append(t)

    # Sort by date after duration assignment
    enhanced_trades.sort(key=lambda x: x['date'])
    return enhanced_trades


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
    for symbol in symbols:
        print(f"Processing symbol: {symbol}")
        daily_df, hourly_df, m15_df = prepare_data_for_symbol(symbol)

        # Run backtest
        bt = BacktestTrailingStrategy(m15_df, initial_capital=CAPITAL, trailing_pct=0.01)
        trades = bt.run()

        # Add duration info to trades
        trades_with_duration = add_trade_durations(trades)

        # Optionally create ML features and train/filter
        # ml_metrics = ml_train_and_filter(m15_df, hourly_df, threshold=0.7)
        ml_metrics = None  # replace with actual call if desired

        # Generate report
        generate_full_backtest_report(
            trades=trades_with_duration,
            price_series=m15_df.set_index('date')['close'],
            initial_capital=CAPITAL,
            strategy_name=f"Trailing Stop Strategy - {symbol}",
            ml_metrics=ml_metrics,
            filename=f"backtest_report_{symbol}.txt",
            commentary="Strategy shows promising returns; consider further validation."
        )


def generate_full_backtest_report(
    trades, price_series, initial_capital,
    strategy_name="Custom Strategy",
    ml_metrics=None,
    ml_feature_importance=None,
    transaction_costs=0.0,
    slippage_impact=0.0,
    commentary="",
    filename="detailed_backtest_report.txt"
):
    df = pd.DataFrame(trades).sort_values('date')
    df_trades = df[df['type'] == 'SELL']

    total_trades = len(df_trades)
    wins = (df_trades['pnl'] > 0).sum()
    losses = total_trades - wins
    win_pct = 100 * wins / total_trades if total_trades > 0 else 0
    net_profit = df_trades['pnl'].sum()
    roi = (net_profit / initial_capital) * 100

    equity = (price_series / price_series.iloc[0]) * initial_capital
    daily_returns = equity.pct_change().dropna()
    trading_days_per_year = 252

    annualized_return = (equity.iloc[-1]/equity.iloc[0])**(trading_days_per_year / len(daily_returns)) - 1

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    sharpe_ratio = np.sqrt(trading_days_per_year) * daily_returns.mean() / daily_returns.std()

    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (np.sqrt(trading_days_per_year) * daily_returns.mean() / downside_returns.std()
                     if not downside_returns.empty else np.nan)

    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = -df_trades[df_trades['pnl'] <= 0]['pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_duration = df_trades['duration'].mean() if 'duration' in df_trades else np.nan

    def max_consecutive(series):
        max_count = count = 0
        last = None
        for v in series:
            is_win = v > 0
            if is_win == last:
                count += 1
            else:
                count = 1
                last = is_win
            if count > max_count:
                max_count = count
        return max_count

    max_consec_wins = max_consecutive(df_trades['pnl'])
    max_consec_losses = max_consecutive(-df_trades['pnl'])

    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = df_trades[df_trades['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0

    # Equity Curve Plot
    plt.figure(figsize=(12,6))
    plt.plot(equity.index, equity.values, label='Equity Curve')
    buy_dates = df[df['type']=='BUY']['date']
    sell_dates = df[df['type']=='SELL']['date']
    plt.scatter(buy_dates, equity.reindex(buy_dates, method='nearest'), marker='^', color='green', label='Buy')
    plt.scatter(sell_dates, equity.reindex(sell_dates, method='nearest'), marker='v', color='red', label='Sell')
    plt.title(f'Equity Curve - {strategy_name}')
    plt.xlabel('Date')
    plt.ylabel('Equity Value')
    plt.legend()
    plt.grid(True)
    plot_filename = filename.replace('.txt', '_equity.png')
    plt.savefig(plot_filename)
    plt.close()

    with open(filename, 'w') as f:
        f.write("=== 1. Summary Overview ===\n")
        f.write(f"Strategy Name: {strategy_name}\n")
        f.write(f"Backtest Period: {price_series.index[0].date()} - {price_series.index[-1].date()}\n")
        f.write(f"Initial Capital: {initial_capital}\n")
        f.write(f"Total Trades Executed: {total_trades}\n")
        f.write(f"Winning Trades (%): {win_pct:.2f}%\n")
        f.write(f"Losing Trades (%): {100 - win_pct:.2f}%\n")
        f.write(f"Net Profit/Loss: {net_profit:.2f}\n")
        f.write(f"Return on Investment (ROI %): {roi:.2f}%\n")
        f.write(f"Annualized Return: {annualized_return*100:.2f}%\n")
        f.write(f"Max Drawdown (%): {max_drawdown:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.3f}\n")
        f.write(f"Sortino Ratio: {sortino_ratio:.3f}\n")
        f.write(f"Profit Factor: {profit_factor:.3f}\n")
        f.write(f"Average Trade Duration (days): {avg_duration:.2f}\n\n")

        f.write("=== 2. Equity Curve Visualization ===\n")
        f.write(f"Equity curve plot saved as: {plot_filename}\n\n")

        f.write("=== 3. Performance Metrics ===\n")
        f.write(f"{'Metric':<20} {'Value':<15} Description\n")
        f.write(f"{'-'*65}\n")
        f.write(f"{'Total Net Profit':<20} {net_profit:<15.2f} Total profit or loss over the period\n")
        f.write(f"{'Annualized Return (%)':<20} {annualized_return*100:<15.2f} Return normalized per year\n")
        f.write(f"{'Max Drawdown (%)':<20} {max_drawdown:<15.2f} Largest peak-to-trough loss\n")
        f.write(f"{'Sharpe Ratio':<20} {sharpe_ratio:<15.2f} Risk-adjusted return\n")
        f.write(f"{'Sortino Ratio':<20} {sortino_ratio:<15.2f} Downside risk risk-adjusted return\n")
        f.write(f"{'Win Rate (%)':<20} {win_pct:<15.2f} Percentage of winning trades\n")
        f.write(f"{'Average Win (%)':<20} {avg_win:<15.2f} Average return on winning trades\n")
        f.write(f"{'Average Loss (%)':<20} {avg_loss:<15.2f} Average loss on losing trades\n")
        f.write(f"{'Profit Factor':<20} {profit_factor:<15.2f} Ratio of gross profit to gross loss\n")
        f.write(f"{'Max Consecutive Wins':<20} {max_consec_wins:<15d} Longest winning streak\n")
        f.write(f"{'Max Consecutive Losses':<20} {max_consec_losses:<15d} Longest losing streak\n")
        f.write(f"{'Avg. Trade Duration':<20} {avg_duration:<15.2f} Average holding time per trade\n\n")

        f.write("=== 4. Trade Log ===\n")
        f.write("Trade # | Entry Date | Exit Date | Entry Price | Exit Price | P&L (%) | Duration (days) | Notes\n")
        for i, trade in enumerate(df_trades.itertuples(), 1):
            entry_trades = df[(df['date'] < trade.date) & (df['type'] == 'BUY')]
            entry_trade = entry_trades.iloc[-1] if not entry_trades.empty else None
            notes = getattr(trade, 'exit_reason', '')
            entry_date = entry_trade.date.date() if entry_trade is not None else 'N/A'
            entry_price = entry_trade.price if entry_trade is not None else 'N/A'
            duration = getattr(trade, 'duration', 'N/A')
            f.write(f"{i:<7} | {entry_date} | {trade.date.date()} | {entry_price:<11} | {trade.price:<10} | {trade.pnl:<8.2f} | {duration:<15} | {notes}\n")
        f.write("\n")

        if ml_metrics:
            f.write("=== 5. Indicator / ML Performance Details ===\n")
            for k,v in ml_metrics.items():
                if k != "model":
                    f.write(f"{k}: {v:.4f}\n")
            f.write("\n")

        if ml_feature_importance:
            f.write("Feature Importance:\n")
            for feat, val in sorted(ml_feature_importance.items(), key=lambda x: -x[1])[:20]:
                f.write(f"{feat}: {val:.5f}\n")
            f.write("\n")

        f.write("=== 6. Risk and Exposure Analysis ===\n")
        f.write(f"Max Drawdown (%): {max_drawdown:.2f}\n")
        f.write(f"Volatility (Std Dev of daily returns): {daily_returns.std():.6f}\n")
        f.write("Exposure, recovery time, and additional risk metrics to be added here.\n\n")

        f.write("=== 7. Transaction Costs and Slippage Impact ===\n")
        f.write(f"Transaction costs deducted: {transaction_costs}\n")
        f.write(f"Slippage impact estimated: {slippage_impact}\n\n")

        f.write("=== 8. Commentary and Insights ===\n")
        f.write(commentary if commentary else "No commentary provided.\n")

    print(f"Detailed backtest report saved to {filename}")
    with open(filename) as f:
        print(f.read())


