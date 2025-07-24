# fetch_intraday_data.py

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import os
import json
import pandas as pd

from credentials import get_kite  # ✅ Make sure this exists
# LARGE_MID_CAP_FILE = "/root/falah-ai-bot/large_mid_cap.json"  # optional

INTRADAY_DIR = "/root/falah-ai-bot/intraday_data/"
TIMEFRAME = "15minute"  # change to "60minute" if needed
DAYS = 5

def get_all_instruments(kite):
    try:
        instruments = kite.instruments("NSE")
        token_map = {inst['tradingsymbol']: inst['instrument_token'] for inst in instruments}
        return token_map
    except Exception as e:
        print(f"❌ Failed to fetch instruments: {e}")
        return {}

def fetch_intraday_data(symbols, interval=TIMEFRAME, days=DAYS):
    kite = get_kite()
    token_map = get_all_instruments(kite)

    if not os.path.exists(INTRADAY_DIR):
        os.makedirs(INTRADAY_DIR)

    for symbol in symbols:
        try:
            token = token_map.get(symbol)
            if not token:
                print(f"⚠️ Token not found for {symbol}")
                continue

            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)

            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )

            if not candles:
                print(f"⚠️ No candles for {symbol}")
                continue

            df = pd.DataFrame(candles)
            df.to_csv(f"{INTRADAY_DIR}/{symbol}.csv", index=False)
            print(f"✅ Saved: {symbol}")
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

if __name__ == "__main__":
    # Static test symbols — you can load dynamically instead
    symbols = ["RELIANCE", "INFY", "TCS"]

    # Optional: load from large_mid_cap.json
    # with open(LARGE_MID_CAP_FILE) as f:
    #     symbols = json.load(f)

    fetch_intraday_data(symbols)
