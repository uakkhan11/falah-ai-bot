# live_price_reader.py

import json
import os
import pandas as pd
from amfi_fetcher import load_large_midcap_symbols

HISTORICAL_DATA_DIR = "/root/falah-ai-bot/historical_data"
LIVE_PRICE_FILE = "live_prices.json"

def get_symbol_price_map():
    price_map = {}

    # Step 1: Try to load live prices from cached file
    if os.path.exists(LIVE_PRICE_FILE):
        try:
            with open(LIVE_PRICE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    print("✅ Using live prices from cache.")
                    return {k: round(float(v), 2) for k, v in data.items()}
        except Exception as e:
            print(f"⚠️ Error reading {LIVE_PRICE_FILE}: {e}")

    # Step 2: Fallback to last close from historical data
    print("⚠️ Live prices not found. Falling back to last close prices from historical data.")
    symbols = load_large_midcap_symbols()

    for symbol in symbols:
        filepath = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    last_close = df["close"].iloc[-1]
                    price_map[symbol] = round(float(last_close), 2)
            except Exception as e:
                print(f"⚠️ Failed to load data for {symbol}: {e}")
        else:
            print(f"⚠️ Historical file missing: {symbol}.csv")

    if not price_map:
        print("❌ No prices available from live or fallback data.")
    else:
        print(f"✅ Fallback prices loaded for {len(price_map)} symbols.")

    return price_map
