import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIG ===
CSV_PATH = "./backtest_results/backtest_trades.csv"
assert os.path.exists(CSV_PATH), f"❌ File not found: {CSV_PATH}"

df = pd.read_csv(CSV_PATH, parse_dates=["date"])

# === STATS ===
total_trades = len(df)
win_rate = (df.pnl > 0).mean() * 100
avg_win = df[df.pnl > 0].pnl.mean()
avg_loss = df[df.pnl < 0].pnl.mean()
net_pnl = df.pnl.sum()
profit_factor = -avg_win / avg_loss if avg_loss else np.nan

# === EQUITY CURVE ===
df = df.sort_values("date")
df["equity"] = df.pnl.cumsum()

# === DRAWDOWN ===
df["peak"] = df.equity.cummax()
df["dd"] = df.equity - df.peak
df["dd_pct"] = df.dd / df.peak
max_dd = df.dd.min()
max_dd_pct = df.dd_pct.min() * 100

# === SHARPE RATIO (daily assumed) ===
daily = df.groupby("date")["pnl"].sum()
daily_ret = daily / 1_000_000  # normalize to initial capital
sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() else 0

# === PRINT REPORT ===
print("\n📊 Backtest Report")
print(f"Total Trades     : {total_trades}")
print(f"Win Rate         : {win_rate:.2f}%")
print(f"Avg Win          : ₹{avg_win:.2f}")
print(f"Avg Loss         : ₹{avg_loss:.2f}")
print(f"Net P&L          : ₹{net_pnl:,.2f}")
print(f"Profit Factor    : {profit_factor:.2f}")
print(f"Max Drawdown     : ₹{max_dd:,.2f} ({max_dd_pct:.2f}%)")
print(f"Sharpe Ratio     : {sharpe:.2f}")

# === PLOTS ===
plt.figure(figsize=(12, 5))
plt.plot(df.date, df.equity, label="Equity Curve")
plt.fill_between(df.date, df.peak, df.equity, color='red', alpha=0.3, label="Drawdown")
plt.title("Equity Curve with Drawdown")
plt.xlabel("Date")
plt.ylabel("₹ Equity")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("./backtest_results/equity_curve.png")
print("📈 Saved equity curve to backtest_results/equity_curve.png")

# === Monthly P&L ===
df["month"] = df.date.dt.to_period("M")
monthly = df.groupby("month")["pnl"].sum()

plt.figure(figsize=(12, 4))
monthly.plot(kind="bar", color="skyblue")
plt.title("Monthly P&L")
plt.ylabel("₹")
plt.grid(True)
plt.tight_layout()
plt.savefig("./backtest_results/monthly_pnl.png")
print("📈 Saved monthly P&L chart to backtest_results/monthly_pnl.png")
