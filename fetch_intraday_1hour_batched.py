import os
import json
import time
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta

DATA_DIR = "/root/falah-ai-bot/historical_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load credentials
with open("/root/falah-ai-bot/secrets.json") as f:
    secrets = json.load(f)

kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
kite.set_access_token(secrets["zerodha"]["access_token"])

with open("/root/falah-ai-bot/tokens.json") as f:
    token_map = json.load(f)

symbols = list(token_map.keys())
print(f"✅ Loaded {len(symbols)} symbols.")

to_date = datetime.today()
from_date = to_date - timedelta(days=730)

MAX_DAYS = 400

for symbol in symbols:
    token = token_map.get(symbol)
    if not token:
        print(f"⚠️ No token for {symbol}")
        continue

    all_batches = []
    start = from_date

    print(f"⬇️ Downloading {symbol}...")

    while start < to_date:
        end = min(start + timedelta(days=MAX_DAYS), to_date)

        try:
            data = kite.historical_data(
                instrument_token=token,
                from_date=start,
                to_date=end,
                interval="60minute"
            )

            if data:
                df = pd.DataFrame(data)
                all_batches.append(df)
                print(f"✅ {symbol}: {len(df)} rows from {start.date()} to {end.date()}")
            else:
                print(f"⚠️ {symbol}: No data from {start.date()} to {end.date()}")

            time.sleep(0.3)

        except Exception as e:
            print(f"❌ {symbol}: {e}")
            break

        start = end + timedelta(days=1)

    if all_batches:
        full_df = pd.concat(all_batches, ignore_index=True).drop_duplicates().sort_values("date")
        outfile = os.path.join(DATA_DIR, f"{symbol}.csv")
        full_df.to_csv(outfile, index=False)
        print(f"✅ {symbol}: Saved {len(full_df)} rows.")
    else:
        print(f"⚠️ {symbol}: No data collected.")

print("✅ All symbols processed.")
