import os
import glob
import pandas as pd
import backtrader as bt
from bt_falah import FalahStrategy

# ─── CONFIG ───────────────────────────────────────────────
RESULTS_DIR = "./backtest_results"
DATA_DIR = "fetch_intraday_1hour_batched.py"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Initialize Cerebro ───────────────────────────────────
cerebro = bt.Cerebro()
cerebro.broker.setcash(1_000_000)
cerebro.broker.setcommission(commission=0.0005)

# ─── Load All Data in 1-hour Timeframe ────────────────────
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in historical_data folder.")

print(f"✅ Found {len(csv_files)} CSV files.")

loaded_files = 0

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    if df.shape[0] < 100:
        print(f"⚠️ {os.path.basename(csv_file)} skipped (too few rows: {df.shape[0]})")
        continue
    if df["close"].nunique() == 1:
        print(f"⚠️ {os.path.basename(csv_file)} skipped (constant price)")
        continue
    if df["close"].isna().all():
        print(f"⚠️ {os.path.basename(csv_file)} skipped (all NaNs)")
        continue

    symbol = os.path.basename(csv_file).replace(".csv", "")
    data = bt.feeds.GenericCSVData(
        dataname=csv_file,
        dtformat="%Y-%m-%d %H:%M:%S%z",
        timeframe=bt.TimeFrame.Minutes,
        compression=60,  # 1-hour bars
        openinterest=-1
    )
    cerebro.adddata(data, name=symbol)
    loaded_files += 1

print(f"✅ Loaded {loaded_files} valid CSV files into Backtrader.")
print("🚀 Starting Backtest...")

# ─── Add Strategy ────────────────────────────────────────
cerebro.addstrategy(FalahStrategy)

# ─── Run Backtest ────────────────────────────────────────
print("Starting Portfolio Value:", cerebro.broker.getvalue())
results = cerebro.run()
print("Ending Portfolio Value:", cerebro.broker.getvalue() )

# ─── Retrieve Data From Strategy ─────────────────────────
strategy_instance = results[0]

trades = getattr(strategy_instance, "trades_log", [])
equity_curve = getattr(strategy_instance, "equity_curve", [])
drawdowns = getattr(strategy_instance, "drawdowns", [])

# ─── Save Results ────────────────────────────────────────
if trades:
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(os.path.join(RESULTS_DIR, "trades.csv"), index=False)

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
            (ec["capital"].iloc[-1] / ec["capital"].iloc[0]) ** (
                1 / ((ec["date"].iloc[-1] - ec["date"].iloc[0]).days / 365.25)
            )
        ) - 1
        sharpe = returns.mean() / returns.std() * (252 ** 0.5)
    else:
        cagr = sharpe = 0

    max_dd = max(drawdowns) * 100 if drawdowns else 0

    print("\n🎯 Backtest Performance:")
    print(f"Final Portfolio Value: ₹{ec['capital'].iloc[-1]:,.2f}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.1f}%")
else:
    print("⚠️ No equity curve data to compute performance metrics.")
