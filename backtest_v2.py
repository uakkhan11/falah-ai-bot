import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_full_backtest_report(
    trades, price_series, initial_capital,
    strategy_name="Custom Strategy",
    ml_metrics=None,
    ml_confusion_matrix=None,
    ml_feature_importance=None,
    transaction_costs=0.0,
    slippage_impact=0.0,
    commentary="",
    filename="detailed_backtest_report.txt"
):
    # Prepare DataFrame from trades
    df = pd.DataFrame(trades)
    df = df.sort_values(by='date').reset_index(drop=True)
    df_trades = df[df['type'] == "SELL"]
    
    # Basic stats
    total_trades = len(df_trades)
    wins = (df_trades['pnl'] > 0).sum()
    losses = total_trades - wins
    win_pct = 100 * wins / total_trades if total_trades > 0 else 0
    net_profit = df_trades['pnl'].sum()
    roi = (net_profit / initial_capital) * 100

    # Calculate equity curve from price series
    equity = (price_series / price_series.iloc[0]) * initial_capital
    daily_returns = equity.pct_change().dropna()
    trading_days_per_year = 252

    total_days = (price_series.index[-1] - price_series.index[0]).days
    annualized_return = (equity.iloc[-1]/equity.iloc[0])**(trading_days_per_year/len(daily_returns)) - 1

    # Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    # Risk metrics
    sharpe_ratio = np.sqrt(trading_days_per_year) * daily_returns.mean() / daily_returns.std()

    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (np.sqrt(trading_days_per_year) * daily_returns.mean() / downside_returns.std()
                     if not downside_returns.empty else np.nan)

    gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    gross_loss = -df_trades[df_trades['pnl'] <= 0]['pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    avg_duration = df_trades['duration'].mean() if 'duration' in df_trades.columns else np.nan

    # Max consecutive wins/losses
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

    # Plot equity curve with buy/sell markers
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

    # Write textual report
    with open(filename, 'w') as f:
        # 1. Overview
        f.write("=== 1. Summary Overview ===\n")
        f.write(f"Strategy Name: {strategy_name}\n")
        f.write(f"Backtest Period: {price_series.index[0].date()} to {price_series.index[-1].date()}\n")
        f.write(f"Initial Capital: {initial_capital}\n")
        f.write(f"Total Trades Executed: {total_trades}\n")
        f.write(f"Winning Trades (%): {win_pct:.2f}%\n")
        f.write(f"Losing Trades (%): {100-win_pct:.2f}%\n")
        f.write(f"Net Profit (INR): {net_profit:.2f}\n")
        f.write(f"Return on Investment (ROI %): {roi:.2f}%\n")
        f.write(f"Annualized Return (%): {annualized_return*100:.2f}%\n")
        f.write(f"Max Drawdown (%): {max_drawdown:.2f}%\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.3f}\n")
        f.write(f"Sortino Ratio: {sortino_ratio:.3f}\n")
        f.write(f"Profit Factor: {profit_factor:.3f}\n")
        f.write(f"Average Trade Duration (Days): {avg_duration:.2f}\n\n")

        # 2. Equity Curve
        f.write("=== 2. Equity Curve Visualization ===\n")
        f.write(f"Equity curve plot saved as: {plot_filename}\n\n")

        # 3. Performance Metrics
        f.write("=== 3. Performance Metrics ===\n")
        f.write(f"{'Metric':<20} {'Value':<15} Description\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'Total Net Profit':<20} {net_profit:<15.2f} Total profit over the period\n")
        f.write(f"{'Annualized Return':<20} {annualized_return*100:<15.2f} Return normalized per year\n")
        f.write(f"{'Max Drawdown':<20} {max_drawdown:<15.2f} Largest peak-to-trough loss\n")
        f.write(f"{'Sharpe Ratio':<20} {sharpe_ratio:<15.2f} Risk-adjusted return\n")
        f.write(f"{'Sortino Ratio':<20} {sortino_ratio:<15.2f} Downside risk-adjusted return\n")
        f.write(f"{'Win Rate (%)':<20} {win_pct:<15.2f} Percentage of winning trades\n")
        f.write(f"{'Average Win (%)':<20} {avg_win:<15.2f} Average profit on winning trades\n")
        f.write(f"{'Average Loss (%)':<20} {avg_loss:<15.2f} Average loss on losing trades\n")
        f.write(f"{'Profit Factor':<20} {profit_factor:<15.2f} Ratio of gross profit to loss\n")
        f.write(f"{'Max Consecutive Wins':<20} {max_consec_wins:<15d} Longest winning streak\n")
        f.write(f"{'Max Consecutive Losses':<20} {max_consec_losses:<15d} Longest losing streak\n")
        f.write(f"{'Avg Trade Duration':<20} {avg_duration:<15.2f} Average holding period in days\n\n")

        # 4. Trade Log
        f.write("=== 4. Trade Log ===\n")
        f.write(f"Trade # | Entry Date  | Exit Date   | Entry Price | Exit Price | PnL (%)    | Duration | Notes\n")
        f.write(f"{'-'*100}\n")
        for idx, trade in enumerate(df_trades.itertuples(), start=1):
            # Match the last buy before sell
            buy_trades = df[(df['date'] < trade.date) & (df['type'] == 'BUY')]
            entry_trade = buy_trades.iloc[-1] if not buy_trades.empty else None
            notes = getattr(trade, 'exit_reason', "")
            entry_date = entry_trade.date.date() if entry_trade is not None else 'N/A'
            entry_price = entry_trade.price if entry_trade is not None else 'N/A'
            duration = getattr(trade, 'duration', 'N/A')
            f.write(f"{idx:<7} | {entry_date} | {trade.date.date()} | {entry_price:<11} | {trade.price:<10} | "
                    f"{trade.pnl:<10.2f} | {duration:<8} | {notes}\n")
        f.write("\n")

        # 5. Indicator & ML performance
        if ml_metrics:
            f.write("=== 5. Indicator / ML Performance ===\n")
            for k,v in ml_metrics.items():
                if k != "model":
                    f.write(f"{k}: {v:.4f}\n")
            f.write("\n")

        if ml_confusion_matrix is not None:
            f.write("Confusion Matrix Summary (ML):\n")
            f.write(str(ml_confusion_matrix) + "\n\n")

        if ml_feature_importance:
            f.write("ML Feature Importance:\n")
            for feat, val in sorted(ml_feature_importance.items(), key=lambda x: -x[1])[:20]:
                f.write(f"{feat}: {val:.5f}\n")
            f.write("\n")

        # 6. Risk and Exposure Analysis
        f.write("=== 6. Risk and Exposure Analysis ===\n")
        f.write(f"Max Drawdown: {max_drawdown:.2f}%\n")
        f.write(f"Volatility (Std Dev of daily returns): {daily_returns.std():.6f}\n")
        f.write(f"Exposure and recovery time: (to be calculated...)\n\n")

        # 7. Transaction Costs & Slippage
        f.write("=== 7. Transaction Costs and Slippage ===\n")
        f.write(f"Total transaction costs deducted: {transaction_costs}\n")
        f.write(f"Estimated impact of slippage: {slippage_impact}\n\n")

        # 8. Commentary and Insights
        f.write("=== 8. Commentary and Insights ===\n")
        f.write(commentary or "No commentary provided.\n")

    print(f"Full backtest report saved to {filename}")
    # Optionally show report on console after generating
    with open(filename) as f:
        print(f.read())
