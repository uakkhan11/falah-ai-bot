import os
import glob
import pandas as pd
import backtrader as bt
from bt_falah import FalahStrategy

# ─── CONFIG ───────────────────────────────────────────────
RESULTS_DIR = "./backtest_results"
DATA_DIR = "/root/falah-ai-bot/historical_data"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Initialize Cerebro ───────────────────────────────────
cerebro = bt.Cerebro()
cerebro.broker.setcash(1_000_000)
cerebro.broker.setcommission(commission=0.0005)

# ─── Load All Data manually ───────────────────────────────
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in historical_data folder.")

print(f"✅ Found {len(csv_files)} CSV files.")

loaded_files = 0

for csv_file in csv_files:
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df["date"] = df["date"].dt.tz_localize(None)

    # Skip empty or bad files
    if df.shape[0] < 100:
        print(f"⚠️ {os.path.basename(csv_file)} skipped (too few rows: {df.shape[0]})")
        continue
    if df["close"].nunique() == 1:
        print(f"⚠️ {os.path.basename(csv_file)} skipped (constant price)")
        continue
    if df["close"].isna().all():
        print(f"⚠️ {os.path.basename(csv_file)} skipped (all NaNs)")
        continue

    df = df.sort_values("date")

    # 🚀 Set date as index (THIS IS THE FIX)
    df = df.set_index("date")

    data = bt.feeds.PandasData(
        dataname=df,
        timeframe=bt.TimeFrame.Minutes,
        compression=60  # 1-hour bars
    )

    symbol = os.path.basename(csv_file).replace(".csv", "")
    cerebro.adddata(data, name=symbol)
    loaded_files += 1


print(f"✅ Loaded {loaded_files} valid DataFrames into Backtrader.")
print("🚀 Starting Backtest...")

# ─── Add Strategy ────────────────────────────────────────
cerebro.addstrategy(FalahStrategy)

print("\n✅ Verifying all feeds before running Backtest...")
for data in cerebro.datas:
    df = data.p.dataname
    if isinstance(df, pd.DataFrame):
        first = df.iloc[0]["date"]
        if not pd.api.types.is_datetime64_any_dtype(pd.Series([first])):
            print("❌ INVALID DATETIME in", data._name, ":", type(first), first)
        else:
            print("✅", data._name, "ok.")


# ─── Run Backtest ────────────────────────────────────────
print("Starting Portfolio Value:", cerebro.broker.getvalue())
results = cerebro.run()
print("Ending Portfolio Value:", cerebro.broker.getvalue())
