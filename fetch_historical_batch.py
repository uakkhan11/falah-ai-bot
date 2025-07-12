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
    print(f"✅ Loaded {len(symbols)} symbols.")

    to_date = datetime.today()
    from_date = to_date - timedelta(days=5*365)   # 5 years
    BATCH_SIZE = 20

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        print(f"\n🚀 Batch {i//BATCH_SIZE + 1}")

        for sym in batch:
            token = token_map.get(sym)
            if not token:
                print(f"⚠️ No token for {sym}")
                continue

            outfile = os.path.join(OUTPUT_DIR, f"{sym}.csv")

            # Always delete old file to force fresh download
            if os.path.exists(outfile):
                os.remove(outfile)
                print(f"🗑️ {sym}: Old file removed.")

            try:
                print(f"⬇️ Downloading {sym} from {from_date.date()} to {to_date.date()}")
                candles = kite.historical_data(
                    instrument_token=token,
                    from_date=from_date,
                    to_date=to_date,
                    interval="day"
                )

                if not candles:
                    print(f"⚠️ No data for {sym}")
                    continue

                df = pd.DataFrame(candles)
                df.to_csv(outfile, index=False)
                print(f"✅ Saved {sym} ({len(df)} rows).")

            except Exception as e:
                print(f"❌ {sym}: {e}")

            time.sleep(0.3)

if __name__ == "__main__":
    fetch_all_historical()
