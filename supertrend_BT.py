import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configuration: Update path to match your directory structure
DATADIR = 'swing_data'  # Change this to your actual path
END_DATE = datetime(2025, 8, 31)
START_DATE = END_DATE - timedelta(days=5*365)

# Check if the data directory exists
if not os.path.exists(DATADIR):  # Corrected variable name
    raise FileNotFoundError(f"Data directory not found: {DATADIR}")

# Load all CSV files in the specified directory
all_files = [f for f in os.listdir(DATADIR) if f.endswith('.csv')]

# Load data for each symbol
def load_data_for_backtest(directory, symbol_file, start_date, end_date):
    filepath = os.path.join(directory, symbol_file)
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    df = df.sort_values('date').reset_index(drop=True)
    return df

# Supertrend indicator
def supertrend(df, period=10, multiplier=1.5):
    df['tr'] = np.max((df['high'] - df['low'],
                      abs(df['high'] - df['close'].shift(1)),
                      abs(df['low'] - df['close'].shift(1))), axis=0)
    df['atr'] = df['tr'].rolling(period).mean()
    hl2 = (df['high'] + df['low']) / 2
    df['upperband'] = hl2 + (multiplier * df['atr'])
    df['lowerband'] = hl2 - (multiplier * df['atr'])
    df['supertrend'] = True
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i-1]:
            df['supertrend'].iloc[i] = True
        elif df['close'].iloc[i] < df['lowerband'].iloc[i-1]:
            df['supertrend'].iloc[i] = False
        else:
            df['supertrend'].iloc[i] = df['supertrend'].iloc[i-1]
    return df

# Backtest strategy with trailing stop loss and pyramiding
def backtest_strategy(df, symbol, period=10, multiplier=1.5):
    df = supertrend(df, period, multiplier)
    df['position'] = 0
    df['pyramid_count'] = 0
    df['entry_price'] = np.nan
    df['stop_loss'] = np.nan
    df['returns'] = np.nan
    df['cumulative_returns'] = 1
    df['profit'] = 0
    df['symbol'] = symbol
    position = 0
    pyramid_count = 0
    trail_stop = np.nan
    entry_price = np.nan

    for i in range(1, len(df)):
        if df['supertrend'].iloc[i] and not df['supertrend'].iloc[i-1] and position == 0:
            position = 1
            df.at[i, 'position'] = 1
            entry_price = df['close'].iloc[i]
            df.at[i, 'entry_price'] = entry_price
            trail_stop = df['lowerband'].iloc[i]
            df.at[i, 'stop_loss'] = trail_stop
            df.at[i, 'profit'] = 0
        elif position == 1 and df['close'].iloc[i] > df['close'].iloc[i-1]:
            pyramid_count += 1
            df.at[i, 'pyramid_count'] = pyramid_count
        if position == 1:
            if not df['supertrend'].iloc[i] or df['close'].iloc[i] <= trail_stop:
                df.at[i, 'position'] = 0
                position = 0
                df.at[i, 'profit'] = df['close'].iloc[i] - entry_price
                pyramid_count = 0
            elif df['lowerband'].iloc[i] > trail_stop:
                trail_stop = df['lowerband'].iloc[i]
                df.at[i, 'stop_loss'] = trail_stop
        else:
            df.at[i, 'profit'] = 0

        df.at[i, 'returns'] = df['profit'].iloc[i]
        df.at[i, 'cumulative_returns'] = df['cumulative_returns'].iloc[i-1] * (1 + df['profit'].iloc[i] / entry_price if entry_price != np.nan and df['profit'].iloc[i] != 0 else 1)

    return df

# Backtest all symbols in folder
combined_results = pd.DataFrame()
summary_results = []

for file in all_files:
    symbol = file.split('.')[0]
    df = load_data_for_backtest(DATADIR, file, START_DATE, END_DATE)
    if df.empty:
        continue
    result = backtest_strategy(df.copy(), symbol, period=10, multiplier=1.5)
    combined_results = pd.concat([combined_results, result], ignore_index=True)

    # Summary for each symbol
    total_profit = result['profit'].sum()
    wins = (result['profit'] > 0).sum()
    losses = (result['profit'] < 0).sum()
    trade_count = result['profit'].replace(0, np.nan).dropna().count()
    win_rate = 0 if trade_count == 0 else (wins / trade_count) * 100
    summary_results.append({
        "Symbol": symbol,
        "Total Trades": trade_count,
        "Total Profit": total_profit,
        "Winning Trades": wins,
        "Losing Trades": losses,
        "Win Rate": f"{win_rate:.2f}%",
        "Cumulative Return": f"{result['cumulative_returns'].iloc[-1] if not result.empty else 1:.2f}"
    })

# Save results
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv('all_symbols_summary.csv', index=False)
combined_results.to_csv('all_symbols_backtest_detailed.csv', index=False)

print("Backtest complete. Summary and detailed results saved.")

import pandas as pd
import numpy as np

# Load your detailed backtest results
df = pd.read_csv('all_symbols_backtest_detailed.csv')

# Define parameters used in the strategy
atr_mult = 1.5  # Replace with actual value used in strategy
adx_threshold = 20  # Replace with actual value used
trail = True  # Replace with actual trailing stop setting
start_capital = 100000  # Replace with your starting capital
symbol_list = df['symbol'].unique()
total_trades = df[df['position'] == 1]['position'].sum()
total_shares = total_trades + df['pyramid_count'].sum()
total_pnl = df['profit'].sum()
end_capital = start_capital + total_pnl
net_pnl = total_pnl

# Winners and losers
winners = df[df['profit'] > 0]['profit']
losers = df[df['profit'] < 0]['profit']
winners_count = len(winners)
losers_count = len(losers)
percent_profitable = (winners_count / total_trades) * 100 if total_trades > 0 else 0
avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
avg_win = winners.mean() if len(winners) > 0 else 0
avg_loss = losers.mean() if len(losers) > 0 else 0
win_loss_ratio = winners_count / losers_count if losers_count > 0 else 0

# Gross profit and loss
gross_profit = winners.sum()
gross_loss = abs(losers.sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

# Best and worst trades
best_trade = winners.max() if len(winners) > 0 else 0
worst_trade = losers.min() if len(losers) > 0 else 0

# Trade duration (if you have entry/exit time in 'date' col, else use avg over sample)
# For simplicity, if not available, use NaNs or estimate based on rows between entries/exits
avg_trade_duration_days = np.nan
median_trade_duration_days = np.nan

# Time in market (fraction of days capital was deployed)
# You can calculate it from position 1 days vs total days in df
time_in_market_pct = (df['position'] == 1).sum() / len(df) * 100

# Maximum drawdown (abs and %)
capital_curve = df['cumulative_returns'].cumprod() * start_capital
max_drawdown_abs = (capital_curve.cummax() - capital_curve).max()
max_drawdown_pct = ((capital_curve.cummax() - capital_curve) / capital_curve.cummax()).max() * 100

# Volatility, Sharpe, Sortino, CAGR, Calmar - annualized (approx)
returns = df['returns'].dropna()
volatility_annualized = returns.std() * np.sqrt(252) * 100  # 252 trading days
sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
sortino = returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
cagr_pct = (end_capital / start_capital) ** (252 / len(df)) - 1
cagr_pct = cagr_pct * 100
calmar = cagr_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0

# Collect all results
detail_report = {
    "atr_mult": atr_mult,
    "adx_threshold": adx_threshold,
    "trail": trail,
    "start_capital": start_capital,
    "end_capital": end_capital,
    "net_pnl": net_pnl,
    "total_trades": total_trades,
    "winners": winners_count,
    "losers": losers_count,
    "percent_profitable": round(percent_profitable, 2),
    "avg_trade_pnl": round(avg_trade_pnl, 2),
    "avg_win": round(avg_win, 2) if not np.isnan(avg_win) else 0,
    "avg_loss": round(avg_loss, 2) if not np.isnan(avg_loss) else 0,
    "win_loss_ratio": round(win_loss_ratio, 2),
    "gross_profit": round(gross_profit, 2),
    "gross_loss": round(gross_loss, 2),
    "profit_factor": round(profit_factor, 2) if not np.isnan(profit_factor) else 0,
    "best_trade": round(best_trade, 2),
    "worst_trade": round(worst_trade, 2),
    "avg_trade_duration_days": avg_trade_duration_days,
    "median_trade_duration_days": median_trade_duration_days,
    "time_in_market_pct": round(time_in_market_pct, 2),
    "max_drawdown_abs": round(max_drawdown_abs, 2),
    "max_drawdown_pct": round(max_drawdown_pct, 2),
    "volatility_annualized": round(volatility_annualized, 2),
    "sharpe": round(sharpe, 2),
    "sortino": round(sortino, 2),
    "cagr_pct": round(cagr_pct, 2),
    "calmar": round(calmar, 2)
}

# Print and export to CSV for analysis
detail_report_df = pd.DataFrame([detail_report])
print(detail_report_df)
detail_report_df.to_csv('detailed_performance_report.csv', index=False)

