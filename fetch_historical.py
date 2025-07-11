import os
import json
import time
import pandas as pd
from kiteconnect import KiteConnect
from utils import load_credentials

# Initialize Kite
secrets = load_credentials()
creds = secrets["zerodha"]

kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(creds["access_token"])
print("DEBUG CREDS:", creds)

# Load tokens.json
with open("/root/falah-ai-bot/tokens.json", "r") as f:
    tokens = json.load(f)

print(f"✅ Loaded {len(tokens)} tokens.")

# Create output directory
output_dir = "/root/falah-ai-bot/data"
os.makedirs(output_dir, exist_ok=True)

# For each symbol, fetch historical data
for symbol, token in tokens.items():
    filename = os.path.join(output_dir, f"{symbol}.csv")
    if os.path.exists(filename):
        print(f"⚠️ {symbol}: File already exists. Skipping.")
        continue

    print(f"⬇️ Downloading {symbol}...")
    try:
        data = kite.historical_data(
            instrument_token=int(token),
            from_date="2024-01-01",
            to_date="2024-12-31",
            interval="day"
        )
        if not data:
            print(f"❌ No data returned for {symbol}. Skipping.")
            continue

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ Saved {symbol} ({len(df)} rows).")

        time.sleep(0.3)  # Respect rate limits

    except Exception as e:
        print(f"❌ Failed to fetch {symbol}: {e}")

print("✅ All symbols processed.")

with open("/root/falah-ai-bot/last_fetch.txt", "w") as f:
    f.write(datetime.now().isoformat())
