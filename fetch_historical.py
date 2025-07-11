import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from utils import load_credentials

# Initialize Kite
secrets = load_credentials()
if "zerodha" not in secrets:
    raise KeyError("❌ 'zerodha' section missing in secrets.json. Please fix your credentials file.")

with open("/root/falah-ai-bot/secrets.json") as f:
    secrets = json.load(f)
creds = secrets["zerodha"]
print("DEBUG CREDS:", creds)

if "api_key" not in creds or "access_token" not in creds:
    raise KeyError("❌ 'api_key' or 'access_token' missing in 'zerodha' credentials.")

kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(creds["access_token"])

# Load tokens.json
with open("/root/falah-ai-bot/tokens.json", "r") as f:
    tokens = json.load(f)

print(f"✅ Loaded {len(tokens)} tokens.")

# Create output directory
output_dir = "/root/falah-ai-bot/data"
os.makedirs(output_dir, exist_ok=True)

# Define date range: last 5 years
to_date = datetime.today()
from_date = to_date - timedelta(days=5 * 365)

# Main loop
for idx, (symbol, token) in enumerate(tokens.items(), 1):
    filename = os.path.join(output_dir, f"{symbol}.csv")
    if os.path.exists(filename):
        print(f"⚠️ [{idx}/{len(tokens)}] {symbol}: File already exists. Skipping.")
        continue

    print(f"⬇️ [{idx}/{len(tokens)}] Downloading {symbol} ({from_date.date()} to {to_date.date()})...")
    try:
        data = kite.historical_data(
            instrument_token=int(token),
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
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
