# fetch_historical_batch.py

import os
import json
import pandas as pd
import time
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from credentials import load_secrets
from utils import get_halal_list

OUTPUT_DIR = "/root/falah-ai-bot/historical_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load credentials
secrets = load_secrets()
creds = secrets["zerodha"]

kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(creds["access_token"])

# Load instrument tokens
with open("/root/falah-ai-bot/tokens.json") as f:
    token_map = json.load(f)

# Get symbols
symbols = get_halal_list("1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")
print(f"✅ Loaded {len(symbols)} symbols.")

# Date range
to_date = datetime.today()
from_date = to_date - timedelta(days=200)

BATCH_SIZE = 30

for i in range(0, len(symbols), BATCH_SIZE):
    batch = symbols[i:i+BATCH_SIZE]
    print(f"🚀 Batch {i//BATCH_SIZE+1}")

    for sym in batch:
        token = token_map.get(sym)
        if not token:
            print(f"⚠️ No token for {sym}")
            continue

        outfile = os.path.join(OUTPUT_DIR, f"{sym}.csv")
        if os.path.exists(outfile):
            print(f"✅ {sym} already exists.")
            continue

        try:
            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            df = pd.DataFrame(candles)
            df.to_csv(outfile, index=False)
            print(f"✅ Saved {sym}")
        except Exception as e:
            print(f"❌ {sym}: {e}")
        time.sleep(0.3)
