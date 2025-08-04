# live_price_reader.py
import os
import json
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
from credentials import load_secrets

HISTORICAL_DATA_DIR = "/root/falah-ai-bot/historical_data"
LIVE_PRICE_FILE = "live_prices.json"

def get_kite():
    secrets = load_secrets()
    creds = secrets["zerodha"]
    kite = KiteConnect(api_key=creds["api_key"])
    kite.set_access_token(creds["access_token"])
    return kite

def market_is_open():
    now = datetime.now().time()
    return now >= datetime.strptime("09:15", "%H:%M").time() and now <= datetime.strptime("15:30", "%H:%M").time()

def get_symbol_price_map(symbols=None, force_live=False):
    """Return dict {symbol: price}"""
    price_map = {}

    # If market is open or force_live → fetch live from Kite
    if force_live or market_is_open():
        kite = get_kite()
        try:
            instruments = [f"NSE:{sym}" for sym in symbols] if symbols else []
            if instruments:
                data = kite.ltp(instruments)
                price_map = {sym.split(":")[1]: round(v["last_price"], 2) for sym, v in data.items()}
                # Save cache
                with open(LIVE_PRICE_FILE, "w") as f:
                    json.dump(price_map, f)
                print(f"✅ Live prices fetched for {len(price_map)} symbols.")
                return price_map
        except Exception as e:
            print(f"⚠️ Live fetch failed: {e}")

    # Else → try cache
    if os.path.exists(LIVE_PRICE_FILE):
        try:
            with open(LIVE_PRICE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    print(f"📦 Using cached live prices for {len(data)} symbols.")
                    return {k: round(float(v), 2) for k, v in data.items()}
        except Exception as e:
            print(f"⚠️ Failed to read live price cache: {e}")

    # Fallback → historical last close
    print("⚠️ No live prices. Falling back to last close from historical data.")
    if not symbols:
        return {}

    for symbol in symbols:
        filepath = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.csv")
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    price_map[symbol] = round(float(df["close"].iloc[-1]), 2)
            except Exception as e:
                print(f"⚠️ Failed to load {symbol} last close: {e}")

    if price_map:
        print(f"✅ Fallback prices loaded for {len(price_map)} symbols.")
    else:
        print("❌ No prices available from any source.")

    return price_map
