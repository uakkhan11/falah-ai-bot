from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import pytz
import pandas as pd

IST = pytz.timezone("Asia/Kolkata")


def get_cnc_holdings(kite: KiteConnect):
    """
    Fetch CNC holdings.
    """
    return kite.holdings()


def get_live_ltp(kite: KiteConnect, symbol: str):
    """
    Fetch LTP for a symbol.
    """
    try:
        full_symbol = f"NSE:{symbol}"
        print(f"Fetching LTP for {full_symbol}")
        data = kite.ltp(full_symbol)
        print("Raw LTP Response:", data)

        if full_symbol not in data:
            raise Exception(f"Symbol '{symbol}' not found in LTP response.")

        return data[full_symbol]["last_price"]

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise Exception(f"LTP fetch error: {e}")


def fetch_historical_candles(kite: KiteConnect, instrument_token: str, interval="day", days=60):
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

    df = pd.DataFrame(candles)
    if df.empty:
        raise ValueError("No historical data returned.")
    df.columns = [c.capitalize() for c in df.columns]
    if len(df) < 20:
        raise ValueError(f"Only {len(df)} rows fetched. Need at least 20 rows for indicators.")
    return df


def fetch_recent_historical(kite: KiteConnect, symbol: str, interval="15minute", days=5):
    """
    Fetch recent historical data for a given NSE symbol (15m default).
    """
    print(f"Fetching recent historical for {symbol}...")
    instrument_token = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
    print(f"Resolved {symbol} -> Token {instrument_token}")

    to_date = datetime.now(IST)
    from_date = to_date - timedelta(days=days)

    candles = kite.historical_data(
        instrument_token,
        from_date,
        to_date,
        interval
    )

    df = pd.DataFrame(candles)
    if df.empty:
        raise ValueError(f"No data for {symbol} - {interval}.")
    df.columns = [c.capitalize() for c in df.columns]
    return df
