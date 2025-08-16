# live_data_manager.py

import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class LiveDataManager:
    def __init__(self, kite: KiteConnect):
        """
        Initializes with an authenticated KiteConnect instance.
        """
        self.kite = kite
        self.instruments = {}
        self.rate_limit_hit = False

    def get_instruments(self):
        try:
            instruments_list = self.kite.instruments("NSE")  # <<--- THIS LINE fetches the data
            self.instruments = {
                item['tradingsymbol']: item['instrument_token']
                for item in instruments_list                    # <<--- instruments_list is now defined
            }
            return self.instruments
        except Exception as e:
            print(f"❌ Error fetching instruments: {e}")
            return {}

    def _retry_api_call(self, func, *args, retries=3, delay=2, **kwargs):
        """
        Retry wrapper for Kite API calls with exponential backoff.
        """
        for attempt in range(1, retries + 1):
            try:
                result = func(*args, **kwargs)
                self.rate_limit_hit = False
                return result
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate limit" in msg:
                    self.rate_limit_hit = True
                    wait_time = delay * attempt
                    print(f"⚠️ Rate limit hit. Waiting {wait_time}s before retry (Attempt {attempt})...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ API call error: {e}")
                    break
        return None

    def get_bulk_current_prices(self, symbols):
        """
        Fetch LTP for multiple symbols in one quote API call.
        """
        try:
            keys = [f"NSE:{sym}" for sym in symbols]
            quotes = self._retry_api_call(self.kite.quote, keys) or {}
            prices = {
                sym: quotes.get(f"NSE:{sym}", {}).get("last_price")
                for sym in symbols
            }
            return prices
        except Exception as e:
            print(f"❌ Error fetching bulk prices: {e}")
            return {}

    def get_historical_data(self, symbol, interval="day", days=None):
        """
        Fetch historical OHLCV data for symbol and interval, with retry/backoff.
        """
        token = self.instruments.get(symbol)
        if not token:
            print(f"❌ Instrument token not found for {symbol}")
            return None

        limits = {"day": 200, "60minute": 60, "15minute": 20}
        if days is None:
            days = limits.get(interval, 200)
        else:
            max_allowed = limits.get(interval, days)
            if days > max_allowed:
                print(f"⚠️ {interval} limited to {max_allowed} days. Adjusting.")
                days = max_allowed

        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days)

        data = self._retry_api_call(
            self.kite.historical_data,
            instrument_token=token,
            from_date=from_dt.strftime("%Y-%m-%d %H:%M:%S"),
            to_date=to_dt.strftime("%Y-%m-%d %H:%M:%S"),
            interval=interval
        )
        if not data:
            return None

        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        return df

    def get_historical_data_parallel(self, symbols, interval="day", days=200, max_workers=8):
        """
        Fetch historical data for multiple symbols in parallel threads.
        """
        def fetch(sym):
            return sym, self.get_historical_data(sym, interval=interval, days=days)

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fetch, sym) for sym in symbols]
            for fut in as_completed(futures):
                sym, df = fut.result()
                results[sym] = df
        return results

    def get_current_price(self, symbol):
        """
        Fetch current LTP for a single symbol.
        """
        token = self.instruments.get(symbol)
        if not token:
            return None
        try:
            quote = self._retry_api_call(self.kite.quote, [f"NSE:{symbol}"])
            return quote.get(f"NSE:{symbol}", {}).get("last_price")
        except Exception as e:
            print(f"❌ Error fetching price for {symbol}: {e}")
            return None
