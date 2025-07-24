import pandas as pd
from datetime import datetime, timedelta
import pytz

def fetch_recent_historical(kite, symbol, days=30, interval="day"):
    """
    Fetch recent historical data for the given symbol using Zerodha's Kite API.
    Default interval: 'day'
    """
    try:
        end_date = datetime.now(pytz.timezone("Asia/Kolkata"))
        start_date = end_date - timedelta(days=days)
        instrument = f"NSE:{symbol}"

        data = kite.historical_data(
            instrument_token=kite.ltp([instrument])[instrument]['instrument_token'],
            from_date=start_date,
            to_date=end_date,
            interval=interval,
            continuous=False
        )
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠️ Error fetching historical for {symbol}: {e}")
        return pd.DataFrame()
