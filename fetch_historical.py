# fetch_historical.py

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
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

    # Load instrument list from Kite
    print("📥 Downloading instrument list...")
    instruments = kite.instruments("NSE")
    instrument_df = pd.DataFrame(instruments)
    token_map = {
        row["tradingsymbol"]: row["instrument_token"]
        for _, row in instrument_df.iterrows()
    }

    # Load halal list
    symbols = get_halal_list("1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")
    print(f"✅ Loaded {len(symbols)} symbols from halal list.")

    to_date = datetime.today()
    from_date = to_date - timedelta(days=5 * 365)  # Last 5 years
    BATCH_SIZE = 20

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        print(f"\n🚀 Batch {i // BATCH_SIZE + 1}")

        for sym in batch:
            token = token_map.get(sym)
            if not token:
                print(f"⚠️ No token for {sym}")
                continue

            outfile = os.path.join(OUTPUT_DIR, f"{sym}.csv")
            if os.path.exists(outfile):
                os.remove(outfile)  # Always overwrite

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
                print(f"❌ Failed {sym}: {e}")

            time.sleep(0.3)  # Rate limiting


# Save NIFTY index data as reference
def save_nifty_index():
    import yfinance as yf
    df = yf.download("^NSEI", start="2023-01-01")
    df.reset_index(inplace=True)
    df.to_csv("/root/falah-ai-bot/historical_data/NIFTY.csv", index=False)
    print("✅ NIFTY.csv saved.")


if __name__ == "__main__":
    fetch_all_historical()
    save_nifty_index()
