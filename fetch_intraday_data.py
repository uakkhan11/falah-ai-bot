# fetch_intraday_data.py
import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect

from credentials import load_secrets

INTRADAY_DIR = "/root/falah-ai-bot/intraday_data"
TIMEFRAME = "15minute"
DAYS = 5
MAX_RETRIES = 3

def get_kite():
    """Load Zerodha credentials and create KiteConnect instance"""
    secrets = load_secrets()
    creds = secrets["zerodha"]
    kite = KiteConnect(api_key=creds["api_key"])
    kite.set_access_token(creds["access_token"])
    return kite

def get_all_instruments(kite):
    """Get NSE instrument tokens for all tradingsymbols"""
    try:
        instruments = kite.instruments("NSE")
        return {inst["tradingsymbol"]: inst["instrument_token"] for inst in instruments}
    except Exception as e:
        print(f"❌ Failed to fetch instruments list: {e}")
        return {}

def fetch_intraday_data(symbols, interval=TIMEFRAME, days=DAYS):
    """Fetch intraday OHLC data and save CSV files"""
    kite = get_kite()
    token_map = get_all_instruments(kite)

    if not os.path.exists(INTRADAY_DIR):
        os.makedirs(INTRADAY_DIR)

    for symbol in symbols:
        if symbol not in token_map:
            print(f"⚠️ Skipped {symbol}: No token found in NSE instrument list.")
            continue

        token = token_map[symbol]
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                candles = kite.historical_data(
                    instrument_token=token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval
                )
                if not candles:
                    print(f"⚠️ No candles for {symbol}")
                    break

                df = pd.DataFrame(candles)

                # Append forming candle with live price
                try:
                    ltp_data = kite.ltp([f"NSE:{symbol}"])
                    ltp = ltp_data[f"NSE:{symbol}"]["last_price"]
                    if not df.empty:
                        df.iloc[-1, df.columns.get_loc("close")] = ltp
                except Exception as e:
                    print(f"⚠️ Could not update {symbol} with live LTP: {e}")

                df.to_csv(f"{INTRADAY_DIR}/{symbol}.csv", index=False)
                print(f"✅ {symbol} - Saved {len(df)} rows")
                success = True
                break
            except Exception as e:
                print(f"❌ Attempt {attempt} failed for {symbol}: {e}")
                time.sleep(1)

        if not success:
            print(f"❌ {symbol} - Failed after {MAX_RETRIES} attempts.")

if __name__ == "__main__":
    # Example usage: fetch data for a predefined list
    try:
        with open("/root/falah-ai-bot/symbol_to_token.json") as f:
            all_symbols = list(json.load(f).keys())
    except Exception as e:
        print(f"❌ Failed to load symbol list: {e}")
        all_symbols = []

    fetch_intraday_data(all_symbols)
