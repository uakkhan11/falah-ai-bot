# fetch_intraday_data.py

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import os
import json
import pandas as pd
from credentials import load_secrets

INTRADAY_DIR = "/root/falah-ai-bot/intraday_data/"
TIMEFRAME = "15minute"  # default
DAYS = 5

def get_kite():
    secrets = load_secrets()
    creds = secrets["zerodha"]

    kite = KiteConnect(api_key=creds["api_key"])
    kite.set_access_token(creds["access_token"])
    return kite

def get_token_map(kite):
    try:
        instruments = kite.instruments("NSE")
        return {inst['tradingsymbol']: inst['instrument_token'] for inst in instruments}
    except Exception as e:
        print(f"❌ Failed to fetch instruments: {e}")
        return {}

def fetch_intraday_data(symbols, interval=TIMEFRAME, days=DAYS):
    kite = get_kite()
    token_map = get_token_map(kite)

    if not os.path.exists(INTRADAY_DIR):
        os.makedirs(INTRADAY_DIR)

    for symbol in symbols:
        try:
            token = token_map.get(symbol)
            if not token:
                print(f"⚠️ Token not found for {symbol}, skipping.")
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
            print(f"✅ Saved: {symbol} ({len(df)} rows)")

        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

if __name__ == "__main__":
    SCREENED_FILE = "/root/falah-ai-bot/final_screened.json"
    symbols = []

    if os.path.exists(SCREENED_FILE):
        with open(SCREENED_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict):
            symbols = list(data.keys())
        elif isinstance(data, list):
            symbols = data
        print(f"🔍 Loaded {len(symbols)} screened symbols.")
    else:
        print(f"❌ Screened file not found: {SCREENED_FILE}")

    if symbols:
        fetch_intraday_data(symbols)
