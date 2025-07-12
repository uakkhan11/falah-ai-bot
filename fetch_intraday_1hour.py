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

# Load all your symbols
symbols = list(token_map.keys())
print(f"✅ Loaded {len(symbols)} symbols.")

# Download last 2 years (you can adjust)
to_date = datetime.today()
from_date = to_date - timedelta(days=730)

for symbol in symbols:
    token = token_map.get(symbol)
    if not token:
        print(f"⚠️ No token for {symbol}")
        continue

    filename = os.path.join(DATA_DIR, f"{symbol}.csv")
    print(f"⬇️ Downloading {symbol}...")

    try:
        data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval="60minute"
        )
        if not data:
            print(f"❌ No data returned for {symbol}. Skipping.")
            continue

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ Saved {symbol} ({len(df)} rows).")

        time.sleep(0.3)  # respect rate limits
    except Exception as e:
        print(f"❌ {symbol}: {e}")

print("✅ All intraday symbols processed.")
