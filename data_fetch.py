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
       full_symbol = f"NSE:{symbol}"
        data = kite.ltp(full_symbol)
        print("Raw LTP Response:", data)

        if full_symbol not in data:
            raise Exception(f"Symbol '{symbol}' not found in LTP response. Double-check NSE symbol spelling.")

        return data[full_symbol]["last_price"]

        except Exception as e:
            import traceback
            traceback.print_exc()
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
