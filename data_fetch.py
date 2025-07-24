# data_fetch.py
import pandas as pd
import datetime
import time

from kiteconnect import KiteConnect

# ⏱ Convert timeframe to seconds
TIMEFRAME_MAP = {
    'day': 24 * 60 * 60,
    '15minute': 15 * 60,
    '5minute': 5 * 60,
    '1minute': 60
}

def get_live_ltp(kite, symbol):
    """
    Fetches the live Last Traded Price (LTP) for a symbol.
    """
    try:
        instrument = f"NSE:{symbol}"
        ltp_data = kite.ltp([instrument])
        return ltp_data[instrument]['last_price']
    except Exception as e:
        print(f"⚠️ Error fetching LTP for {symbol}: {e}")
        return None


def get_intraday_data(kite, symbol, interval="15minute", days=5):
    """
    Fetches intraday historical data for a symbol for given days and interval.
    """
    try:
        instrument = f"NSE:{symbol}"
        to_date = datetime.datetime.now()
        from_date = to_date - datetime.timedelta(days=days)

        data = kite.historical_data(
            instrument_token=kite.ltp([instrument])[instrument]["instrument_token"],
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        return pd.DataFrame(data)
    except Exception as e:
        print(f"⚠️ Error fetching intraday data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_recent_historical(kite, symbol, days=30):
    """
    Fetch daily candles for recent historical data.
    """
    try:
        instrument = f"NSE:{symbol}"
        to_date = datetime.datetime.now()
        from_date = to_date - datetime.timedelta(days=days)

        data = kite.historical_data(
            instrument_token=kite.ltp([instrument])[instrument]["instrument_token"],
            from_date=from_date,
            to_date=to_date,
            interval="day"
        )
        return pd.DataFrame(data)
    except Exception as e:
        print(f"⚠️ Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_all_historical(kite, symbol_list, days=60, interval="day", output_dir="historical_data"):
    """
    Batch download historical data for multiple symbols and save as CSVs.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for symbol in symbol_list:
        try:
            df = fetch_recent_historical(kite, symbol, days=days)
            if not df.empty:
                df.to_csv(f"{output_dir}/{symbol}.csv", index=False)
                print(f"✅ Saved historical data: {symbol}")
            else:
                print(f"⚠️ No data for {symbol}")
            time.sleep(1)  # Avoid rate limit
        except Exception as e:
            print(f"❌ Error fetching/saving {symbol}: {e}")
