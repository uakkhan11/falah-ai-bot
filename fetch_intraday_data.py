# fetch_intraday_data.py

import os
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from credentials import get_kite, get_all_instruments

INTRADAY_DIR = "/root/falah-ai-bot/intraday_data/"
TIMEFRAME = "15minute"  # change to "60minute" if needed

def fetch_intraday_data(symbols, interval=TIMEFRAME, days=5):
    kite = get_kite()

    if not os.path.exists(INTRADAY_DIR):
        os.makedirs(INTRADAY_DIR)

    for symbol in symbols:
        try:
            token = get_all_instruments()[symbol]
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)

            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )

            df = pd.DataFrame(candles)
            df.to_csv(f"{INTRADAY_DIR}/{symbol}.csv", index=False)
            print(f"✅ Saved: {symbol}")
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")

if __name__ == "__main__":
    # Example symbols (or use from large_mid_cap.json)
    symbols = ["RELIANCE", "INFY", "TCS"]
    fetch_intraday_data(symbols)
