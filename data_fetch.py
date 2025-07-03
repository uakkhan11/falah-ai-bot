# data_fetch.py
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")

def get_cnc_holdings(kite: KiteConnect):
    """
    Fetch CNC holdings.
    """
    positions = kite.holdings()
    return positions

def get_live_ltp(kite: KiteConnect, symbol: str):
    """
    Fetch LTP for a symbol.
    """
    try:
        data = kite.ltp(f"NSE:{symbol}")
        return data[f"NSE:{symbol}"]["last_price"]
    except Exception as e:
        raise Exception(f"LTP fetch error: {e}")

def fetch_historical_candles(kite: KiteConnect, instrument_token: str, interval="15minute", days=5):
    """
    Fetch historical candles for a given instrument.
    """
    to_date = datetime.now(IST)
    from_date = to_date - timedelta(days=days)

    candles = kite.historical_data(
        instrument_token,
        from_date,
        to_date,
        interval
    )
    return candles
