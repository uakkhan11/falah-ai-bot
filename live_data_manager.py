# live_data_manager.py

import pandas as pd
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed


class LiveDataManager:
    def __init__(self, kite: KiteConnect):
        self.kite = kite
        self.instruments = {}

    def get_bulk_current_prices(self, symbols):
        """
        Fetch the latest market prices (LTP) for multiple symbols in a single API call.

        Args:
            symbols (list of str): List of trading symbols, e.g. ["RELIANCE", "TCS"]

        Returns:
            dict: Mapping of symbol -> last_price or None if unavailable
        """
        try:
            instruments = [f"NSE:{sym}" for sym in symbols]
            quotes = self.kite.quote(instruments)

            prices = {}
            for sym in symbols:
                key = f"NSE:{sym}"
                if key in quotes and 'last_price' in quotes[key]:
                    prices[sym] = quotes[key]['last_price']
                else:
                    prices[sym] = None
            return prices
        except Exception as e:
            print(f"Error fetching bulk prices: {e}")
            return {}

    def get_instruments(self):
        """
        Download and cache all NSE instruments for symbol -> instrument_token mapping.

        Returns:
            dict: mapping trading symbol to instrument_token
        """
        try:
            instruments = self.kite.instruments("NSE")
            self.instruments = {item['tradingsymbol']: item['instrument_token'] for item in instruments}
            return self.instruments
        except Exception as e:
            print(f"Error fetching instruments: {e}")
            return {}

    def get_historical_data(self, symbol, interval="day", days=None):
        """
        Fetch historical OHLCV candle data for the given symbol and interval.

        Args:
            symbol (str): Trading symbol, e.g. 'RELIANCE'
            interval (str): "day", "15minute", "60minute"
            days (int or None): Number of days of historical data. Defaults based on interval.

        Returns:
            pandas.DataFrame: DataFrame with OHLCV data sorted by date or None on error
        """
        token = self.instruments.get(symbol)
        if not token:
            print(f"❌ Instrument token not found for {symbol}")
            return None

        # Zerodha API max limits for historical data days per interval
        interval_limits = {
            "day": 200,
            "60minute": 60,
            "15minute": 20
        }

        if days is None:
            days = interval_limits.get(interval, 200)
        else:
            max_allowed = interval_limits.get(interval, days)
            if days > max_allowed:
                print(f"⚠️ {interval} data limited to {max_allowed} days by Zerodha API, adjusting days.")
                days = max_allowed

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        try:
            data = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval
            )
        except Exception as e:
            print(f"Error fetching {interval} data for {symbol}: {e}")
            return None

        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        return df

    def get_current_price(self, symbol):
        """
        Fetch the current market price (LTP) for a single symbol.

        Args:
            symbol (str): Trading symbol, e.g. 'RELIANCE'

        Returns:
            float or None: last traded price if available
        """
        token = self.instruments.get(symbol)
        if token:
            try:
                quote = self.kite.quote(f"NSE:{symbol}")
                return quote[f"NSE:{symbol}"]["last_price"]
            except Exception as e:
                print(f"Error fetching current price for {symbol}: {e}")
                return None
        return None

    def get_historical_data_parallel(self, symbols, interval="day", days=200, max_workers=8):
        """
        Fetch historical OHLCV data for multiple symbols in parallel threads.

        Args:
            symbols (list of str): List of trading symbols
            interval (str): Interval like "day", "15minute", "60minute"
            days (int): Number of days to fetch
            max_workers (int): Number of concurrent threads

        Returns:
            dict: {symbol: DataFrame or None}
        """
        def fetch_symbol_hist(symbol):
            df = self.get_historical_data(symbol, interval=interval, days=days)
            return symbol, df

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_symbol_hist, sym) for sym in symbols]
            for future in as_completed(futures):
                sym, df = future.result()
                results[sym] = df
        return results
