import time
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

class LiveDataManager:
    def __init__(self, kite):
        self.kite = kite
        self.instruments = {}

    def _retry_api_call(self, func, *args, retries=3, delay=2, **kwargs):
        """
        Retry wrapper for Kite API calls with exponential backoff.
        """
        for attempt in range(1, retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait_time = delay * attempt
                    print(f"⚠️ Rate limit hit. Waiting {wait_time}s before retry... (Attempt {attempt})")
                    time.sleep(wait_time)
                else:
                    print(f"Error in API call: {e}")
                    break
        return None

    def get_bulk_current_prices(self, symbols):
        try:
            instruments = [f"NSE:{sym}" for sym in symbols]
            quotes = self._retry_api_call(self.kite.quote, instruments)
            prices = {}
            if quotes:
                for sym in symbols:
                    key = f"NSE:{sym}"
                    prices[sym] = quotes.get(key, {}).get("last_price")
            return prices
        except Exception as e:
            print(f"Error fetching bulk prices: {e}")
            return {}

    def get_historical_data(self, symbol, interval="day", days=None):
        token = self.instruments.get(symbol)
        if not token:
            print(f"❌ No instrument token for {symbol}")
            return None

        interval_limits = {"day": 200, "60minute": 60, "15minute": 20}
        if days is None:
            days = interval_limits.get(interval, 200)
        else:
            max_allowed = interval_limits.get(interval, days)
            if days > max_allowed:
                print(f"⚠️ {interval} limited to {max_allowed} days. Adjusting.")
                days = max_allowed

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        data = self._retry_api_call(
            self.kite.historical_data,
            instrument_token=token,
            from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
            to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
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
        results = {}
        def fetch(sym):
            return sym, self.get_historical_data(sym, interval=interval, days=days)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch, sym) for sym in symbols]
            for future in as_completed(futures):
                sym, df = future.result()
                results[sym] = df
        return results
