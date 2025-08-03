# data_fetch.py

import os
import time
import json
import datetime
import pandas as pd
from kiteconnect import KiteConnect

def get_instrument_token(kite, symbol):
    try:
        with open("symbol_to_token.json") as f:
            token_map = json.load(f)
        return int(token_map.get(symbol))
    except Exception as e:
        print(f"⚠️ Error loading token for {symbol}: {e}")
        return None

def get_live_ltp(kite, symbol):
    try:
        instrument = f"NSE:{symbol}"
        ltp_data = kite.ltp([instrument])
        return ltp_data[instrument]['last_price']
    except Exception as e:
        print(f"⚠️ Error fetching LTP for {symbol}: {e}")
        return None

def get_intraday_data(kite, symbol, interval="15minute", days=1):
    try:
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()

        # Load token map
        with open("token_map.json") as f:
            token_map = json.load(f)

        token = token_map.get(symbol)
        if not token:
            print(f"❌ Token not found for {symbol}")
            return None

        data = kite.historical_data(token, from_date, to_date, interval)

        if not data:
            print(f"⚠️ No data returned for {symbol}")
            return None

        df = pd.DataFrame(data)
        print(f"✅ Data fetched for {symbol}: {len(df)} rows")
        return df

    except Exception as e:
        print(f"⚠️ Error fetching data for {symbol}: {e}")
        return None

def fetch_historical_candles(kite, symbol, interval="day", days=30):
    try:
        token = get_instrument_token(kite, symbol)
        if not token:
            print(f"❌ Skipping {symbol}, instrument token not found.")
            return pd.DataFrame()

        to_date = datetime.datetime.now()
        from_date = to_date - datetime.timedelta(days=days)

        data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )

        df = pd.DataFrame(data)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            print(f"⚠️ 'date' column missing or empty DataFrame for {symbol}")
            return pd.DataFrame()
        return df

    except Exception as e:
        print(f"⚠️ Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_recent_historical(kite, symbol, days=60):
    return fetch_historical_candles(kite, symbol, interval="day", days=days)

def fetch_all_historical(kite, symbol_list, days=60, interval="day", output_dir="historical_data"):
    os.makedirs(output_dir, exist_ok=True)

    for symbol in symbol_list:
        try:
            df = fetch_recent_historical(kite, symbol, days=days)
            if not df.empty:
                filepath = os.path.join(output_dir, f"{symbol}.csv")
                df.to_csv(filepath, index=False)
                print(f"✅ Saved: {symbol} → {filepath}")
            else:
                print(f"⚠️ No data for {symbol}")
            time.sleep(1)  # To avoid rate limits
        except Exception as e:
            print(f"❌ Error for {symbol}: {e}")
