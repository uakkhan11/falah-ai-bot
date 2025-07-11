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

def fetch_all_historical():
    # Load credentials
    secrets = load_secrets()
    creds = secrets["zerodha"]

    kite = KiteConnect(api_key=creds["api_key"])
    kite.set_access_token(creds["access_token"])

    with open("/root/falah-ai-bot/tokens.json") as f:
        token_map = json.load(f)

    symbols = get_halal_list("1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")
    print(f"✅ Loaded {len(symbols)} symbols from your halal list.")

    # Date range: last 5 years
    to_date = datetime.today()
    from_date = to_date - timedelta(days=5 * 365)

    BATCH_SIZE = 30

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        print(f"\n🚀 Batch {i//BATCH_SIZE+1} of {((len(symbols)-1)//BATCH_SIZE)+1}")

        for sym in batch:
            token = token_map.get(sym)
            if not token:
                print(f"⚠️ No token for {sym}. Skipping.")
                continue

            outfile = os.path.join(OUTPUT_DIR, f"{sym}.csv")
            if os.path.exists(outfile):
                print(f"✅ {sym}: Already downloaded. Skipping.")
                continue

            print(f"⬇️ Fetching {sym} ({from_date.date()} to {to_date.date()})...")
            try:
                candles = kite.historical_data(
                    instrument_token=int(token),
                    from_date=from_date.strftime("%Y-%m-%d"),
                    to_date=to_date.strftime("%Y-%m-%d"),
                    interval="day"
                )

                if not candles:
                    print(f"❌ {sym}: No data returned.")
                    continue

                df = pd.DataFrame(candles)
                df.to_csv(outfile, index=False)
                print(f"✅ Saved {sym}: {len(df)} rows.")

            except Exception as e:
                print(f"❌ {sym}: {e}")

            time.sleep(0.3)  # Be polite to the API

if __name__ == "__main__":
    fetch_all_historical()
