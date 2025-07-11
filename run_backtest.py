import os
import pandas as pd
import backtrader as bt
from bt_falah import FalahStrategy

# ─── CONFIG ───────────────────────────────────────────────
RESULTS_DIR = "./backtest_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Initialize Cerebro ───────────────────────────────────
cerebro = bt.Cerebro()
cerebro.broker.setcash(1_000_000)
cerebro.broker.setcommission(commission=0.0005)

# ─── Load Data ────────────────────────────────────────────
data = bt.feeds.GenericCSVData(
    dataname="/root/falah-ai-bot/historical_data/RELIANCE.csv",
    dtformat="%Y-%m-%d %H:%M:%S%z",
    timeframe=bt.TimeFrame.Days,
    compression=1,
    openinterest=-1
)

cerebro.adddata(data)

# ─── Add Strategy ────────────────────────────────────────
cerebro.addstrategy(FalahStrategy)

# ─── Run Backtest ────────────────────────────────────────
print("Starting Portfolio Value:", cerebro.broker.getvalue())
results = cerebro.run()
print("Ending Portfolio Value:", cerebro.broker.getvalue())

# ─── Retrieve Data From Strategy ─────────────────────────
strategy_instance = results[0]

trades = getattr(strategy_instance, "trades_log", [])
equity_curve = getattr(strategy_instance, "equity_curve", [])
drawdowns = getattr(strategy_instance, "drawdowns", [])

# ─── Save Results ────────────────────────────────────────
if trades:
    trades_df = pd.DataFrame(trades)

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
        returns = ec["value"].pct_change().dropna()
        cagr = (
            (ec['value'].iloc[-1] / ec['value'].iloc[0]) ** (
                1 / ((ec['date'].iloc[-1] - ec['date'].iloc[0]).days / 365.25)
            )
        ) - 1
        sharpe = returns.mean() / returns.std() * (252 ** 0.5)
    else:
        cagr = sharpe = 0

    max_dd = max(drawdowns) * 100 if drawdowns else 0

    print("\n🎯 Backtest Performance:")
    print(f"Final Portfolio Value: ₹{ec['value'].iloc[-1]:,.2f}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.1f}%")
else:
    print("⚠️ No equity curve data to compute performance metrics.")

# ─── Show Date Range of CSV ──────────────────────────────
df = pd.read_csv("/root/falah-ai-bot/historical_data/NIFTY.csv", parse_dates=["date"])
print("\n🗓️ Data Range:")
print("First date:", df["date"].min())
print("Last date:", df["date"].max())
print("Total rows:", len(df))
