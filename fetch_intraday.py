import os
import time
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from credentials import load_secrets, get_kite, validate_kite

# Config
DATA_DIR = "/root/falah-ai-bot/intraday_data/"
INTERVALS = ["15minute", "60minute"]
DAYS_BACK = 5
WAIT_TIME = 1  # seconds between API calls

# Make sure the folder exists
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_intraday_data(kite, symbol, interval):
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=DAYS_BACK)
        data = kite.historical_data(
            instrument_token=kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"],
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=False
        )
        df = pd.DataFrame(data)
        if df.empty:
            print(f"⚠️ No data for {symbol} ({interval})")
            return
        df["date"] = pd.to_datetime(df["date"])
        filename = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
        df.to_csv(filename, index=False)
        print(f"✅ Saved {symbol} ({interval})")
    except Exception as e:
        print(f"❌ Failed {symbol} ({interval}): {e}")

def fetch_all_intraday(symbols):
    kite = get_kite()
    if not validate_kite(kite):
        print("❌ Invalid Kite session. Please login.")
        return

    for symbol in symbols:
        for interval in INTERVALS:
            fetch_intraday_data(kite, symbol, interval)
            time.sleep(WAIT_TIME)  # avoid rate limits

if __name__ == "__main__":
    # Example: load from large_mid_cap.json
    try:
        with open("large_mid_cap.json", "r") as f:
            symbols = sorted(set(json.load(f)))
    except:
        symbols = ["TCS", "INFY", "RELIANCE", "HDFCBANK"]  # fallback list

    fetch_all_intraday(symbols)
