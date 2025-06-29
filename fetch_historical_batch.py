import os
import json
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta

# Config
OUTPUT_DIR = "/root/falah-ai-bot/historical_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load credentials
from utils import load_credentials
secrets = load_credentials()
creds = secrets["zerodha"]

# Kite Connect
kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(creds["access_token"])

# Load tokens.json
with open("/root/falah-ai-bot/tokens.json", "r") as f:
    token_map = json.load(f)

# Load HalalList
import gspread
gc = gspread.service_account(filename="/root/falah-credentials.json")
sheet = gc.open_by_key(secrets["sheets"]["SPREADSHEET_KEY"])
symbols = sheet.worksheet("HalalList").col_values(1)[1:]
symbols = [s.strip() for s in symbols if s.strip()]
print(f"✅ Loaded {len(symbols)} symbols from HalalList.")

# Dates
to_date = datetime.today()
from_date = to_date - timedelta(days=200)

# Batch processing
BATCH_SIZE = 50
for i in range(0, len(symbols), BATCH_SIZE):
    batch = symbols[i:i+BATCH_SIZE]
    print(f"🚀 Processing batch {i//BATCH_SIZE +1}: {batch}")

    for sym in batch:
        token = token_map.get(sym)
        if not token:
            print(f"⚠️ No token for {sym}. Skipping.")
            continue

        outfile = os.path.join(OUTPUT_DIR, f"{sym}.csv")
        if os.path.exists(outfile):
            print(f"✅ {sym} already downloaded.")
            continue

        try:
            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            if not candles:
                print(f"⚠️ No candles for {sym}.")
                continue
            df = pd.DataFrame(candles)
            df.to_csv(outfile, index=False)
            print(f"✅ Saved {sym} ({len(df)} rows).")
        except Exception as e:
            print(f"❌ Failed to fetch {sym}: {e}")
