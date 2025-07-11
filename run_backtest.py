import backtrader as bt
from bt_falah import FalahStrategy

trades = []
equity_curve = []
drawdowns = []
capital = INITIAL_CAPITAL


cerebro = bt.Cerebro()
cerebro.broker.setcash(1_000_000)
cerebro.broker.setcommission(commission=0.0005)

data = bt.feeds.GenericCSVData(
    dataname="/root/falah-ai-bot/historical_data/NIFTY.csv",
    dtformat="%Y-%m-%d %H:%M:%S%z",
    timeframe=bt.TimeFrame.Days,
    compression=1,
    openinterest=-1
)

cerebro.adddata(data)
cerebro.addstrategy(FalahStrategy)
print("Starting Portfolio Value:", cerebro.broker.getvalue())
cerebro.run()
print("Ending Portfolio Value:", cerebro.broker.getvalue())
#cerebro.plot()

# ─── SAVE RESULTS ────────────────────────────────────────────────────
if 'trades' not in locals():
    trades = []
if 'equity_curve' not in locals():
    equity_curve = []
if 'drawdowns' not in locals():
    drawdowns = []

if trades:
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(os.path.join(RESULTS_DIR, "trades.csv"), index=False)
    print(f"\n✅ Saved {len(trades)} trades to backtest_results/trades.csv")

    # Summarize trades
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    total_pnl = trades_df["pnl"].sum()
    avg_pnl = trades_df["pnl"].mean()
    win_rate = len(wins) / len(trades_df) * 100

    print("\n📊 Trade Summary:")
    print(f"Total Trades: {len(trades_df)}")
    print(f"Winning Trades: {len(wins)} ({win_rate:.1f}%)")
    print(f"Losing Trades: {len(losses)}")
    print(f"Net P&L: ₹{total_pnl:,.2f}")
    print(f"Average P&L per Trade: ₹{avg_pnl:,.2f}")

else:
    print("\n⚠️ No trades recorded. Nothing to report.")

if equity_curve:
    ec = pd.DataFrame(equity_curve)
    ec.to_csv(os.path.join(RESULTS_DIR, "equity_curve.csv"), index=False)
    if len(ec) > 1:
        returns = ec["capital"].pct_change().dropna()
        cagr = (
            (capital / INITIAL_CAPITAL) ** (1 / ((ec['date'].iloc[-1] - ec['date'].iloc[0]).days / 365.25))
        ) - 1
        sharpe = returns.mean() / returns.std() * (252 ** 0.5)
    else:
        cagr = sharpe = 0

    max_dd = max(drawdowns) * 100 if drawdowns else 0

    print("\n🎯 Backtest Performance:")
    print(f"Final Portfolio Value: ₹{capital:,.2f}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.1f}%")

else:
    print("⚠️ No equity curve data to compute performance metrics.")
