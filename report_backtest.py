import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")

def generate_backtest_report(
    trades, price_series, initial_capital,
    ml_metrics=None, ml_feature_importance=None,
    transaction_costs=0.0, slippage_impact=0.0,
    commentary=None,
    filename="backtest_report.txt"
):
    df = pd.DataFrame(trades)
    df_closed = df[df['type'] == 'SELL']

    # Summary Overview
    total_trades = len(df_closed)
    wins = (df_closed['pnl'] > 0).sum()
    losses = total_trades - wins
    win_pct = 100 * wins / total_trades if total_trades > 0 else 0
    net_profit = df_closed['pnl'].sum()
    roi = (net_profit / initial_capital) * 100

    equity = (price_series / price_series.iloc[0]) * initial_capital
    daily_returns = equity.pct_change().dropna()
    trading_days_per_year = 252

    annualized_return = (equity.iloc[-1] / equity.iloc[0]) ** (trading_days_per_year / len(daily_returns)) - 1
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max)/rolling_max
    max_drawdown = drawdown.min() * 100

    sharpe_ratio = np.sqrt(trading_days_per_year) * daily_returns.mean() / daily_returns.std()
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (np.sqrt(trading_days_per_year) * daily_returns.mean() / downside_returns.std()
                     if not downside_returns.empty else np.nan)

    gross_profit = df_closed[df_closed['pnl'] > 0]['pnl'].sum()
    gross_loss = -df_closed[df_closed['pnl'] <= 0]['pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_duration = df_closed['duration'].mean() if 'duration' in df_closed else np.nan

    def max_consecutive(series):
        max_count = count = 0
        for res in series:
            if res:
                count += 1
                max_count = max(count, max_count)
            else:
                count = 0
        return max_count

    max_consec_wins = max_consecutive(df_closed['pnl'] > 0)
    max_consec_losses = max_consecutive(df_closed['pnl'] <= 0)

    avg_win = df_closed[df_closed['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
    avg_loss = df_closed[df_closed['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0

    # Equity Curve Visualization
    plt.figure(figsize=(10,6))
    plt.plot(equity.index, equity.values, label="Equity Curve")
    buy_dates = df[df['type']=='BUY']['date']
    sell_dates = df[df['type']=='SELL']['date']
    plt.scatter(buy_dates, equity.reindex(buy_dates, method='nearest'), marker='^', color='g', label='Buy')
    plt.scatter(sell_dates, equity.reindex(sell_dates, method='nearest'), marker='v', color='r', label='Sell')
    plt.title('Equity Curve with Trade Markers')
    plt.legend()
    plt.grid(True)
    plot_filename = filename.replace('.txt', '_equity.png')
    plt.savefig(plot_filename)
    plt.close()

    with open(filename, "w") as f:
        # 1. Summary Overview
        f.write("=== 1. Summary Overview ===\n")
        f.write(f"Strategy Name: Custom Strategy\n")
        f.write(f"Backtest Period: {price_series.index[0].date()} - {price_series.index[-1].date()}\n")
        f.write(f"Initial Capital: {initial_capital}\n")
        f.write(f"Total Trades Executed: {total_trades}\n")
        f.write(f"Winning Trades (%): {win_pct:.2f}%\n")
        f.write(f"Losing Trades (%): {100-win_pct:.2f}%\n")
        f.write(f"Net Profit/Loss: {net_profit:.2f}\n")
        f.write(f"Return on Investment (ROI %): {roi:.2f}%\n")
        f.write(f"Annualized Return: {annualized_return*100:.2f}%\n")
        f.write(f"Max Drawdown (%): {max_drawdown:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
        f.write(f"Sortino Ratio: {sortino_ratio:.4f}\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n")
        f.write(f"Average Trade Duration (days): {avg_duration:.2f}\n\n")

        # 2. Equity Curve
        f.write("=== 2. Equity Curve Visualization ===\n")
        f.write(f"Equity curve plot saved as: {plot_filename}\n\n")

        # 3. Performance Metrics
        f.write("=== 3. Performance Metrics ===\n")
        f.write(f"Total Net Profit: {net_profit:.2f}\n")
        f.write(f"Annualized Return (%): {annualized_return*100:.2f}\n")
        f.write(f"Max Drawdown (%): {max_drawdown:.2f}\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
        f.write(f"Sortino Ratio: {sortino_ratio:.4f}\n")
        f.write(f"Win Rate (%): {win_pct:.2f}\n")
        f.write(f"Average Win (%): {avg_win:.2f}\n")
        f.write(f"Average Loss (%): {avg_loss:.2f}\n")
        f.write(f"Profit Factor: {profit_factor:.2f}\n")
        f.write(f"Max Consecutive Wins: {max_consec_wins}\n")
        f.write(f"Max Consecutive Losses: {max_consec_losses}\n")
        f.write(f"Avg. Trade Duration: {avg_duration:.2f}\n\n")

        # 4. Trade Log
        f.write("=== 4. Trade Log ===\n")
        f.write("Trade# | Entry Date | Exit Date | Entry Price | Exit Price | PnL (%) | Duration (days) | Notes\n")
        for i, trade in enumerate(df_closed.itertuples(), start=1):
            # find the entry trade matching exit
            entry_trades = df[(df['date'] < trade.date) & (df['type'] == 'BUY')]
            entry_trade = entry_trades.iloc[-1] if not entry_trades.empty else None
            notes = getattr(trade, 'exit_reason', '')
            f.write(f"{i} | {entry_trade['date'].date() if entry_trade is not None else 'N/A'} | {trade.date.date()} | "
                    f"{entry_trade['price'] if entry_trade is not None else 'N/A'} | {trade.price} | {trade.pnl:.2f} | "
                    f"{getattr(trade, 'duration', 'N/A')} | {notes}\n")
        f.write("\n")

        # 5. Indicator / ML Performance Details
        if ml_metrics is not None:
            f.write("=== 5. Indicator / ML Performance Details ===\n")
            for k, v in ml_metrics.items():
                if k != "model":
                    f.write(f"{k}: {v:.4f}\n")
            f.write("\n")

        if ml_feature_importance is not None:
            f.write("Feature Importance:\n")
            for feat, imp in ml_feature_importance.items():
                f.write(f"{feat}: {imp}\n")
            f.write("\n")

        # 6. Risk and Exposure Analysis (extend with your data)
        f.write("=== 6. Risk and Exposure Analysis ===\n")
        f.write(f"Max Drawdown (%) : {max_drawdown:.2f}\n")
        f.write(f"Volatility (std dev of daily returns): {daily_returns.std():.4f}\n")
        f.write("Exposure and recovery time calculations can be added here.\n\n")

        # 7. Transaction Costs and Slippage Impact (extend with your data)
        f.write("=== 7. Transaction Costs and Slippage Impact ===\n")
        f.write(f"Transaction Costs Deducted: {transaction_costs}\n")
        f.write(f"Slippage Impact on Net Returns: {slippage_impact}\n\n")

        # 8. Commentary and Insights
        f.write("=== 8. Commentary and Insights ===\n")
        f.write(commentary if commentary is not None else "Add commentary and insights here.\n")

    # Print report on terminal
    with open(filename, 'r') as f:
        print(f.read())

# Example call after your backtest (replace arguments with your real data):
# generate_backtest_report(trades_list, price_series, CAPITAL, ml_metrics=ml_results, ml_feature_importance=feature_imp_dict)
