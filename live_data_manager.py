# live_data_manager.py

import pandas as pd
import time
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import pandas_ta as ta

class LiveDataManager:
    def get_bulk_current_prices(self, symbols):
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
        """Download and cache instrument list"""
        instruments = self.kite.instruments("NSE")
        self.instruments = {item['tradingsymbol']: item['instrument_token'] 
                           for item in instruments}
        return self.instruments
    
    class LiveDataManager:
    def __init__(self, kite):
        self.kite = kite
        self.instruments = {}

    def get_instruments(self):
        """Download and cache NSE instrument list"""
        instruments = self.kite.instruments("NSE")
        self.instruments = {item['tradingsymbol']: item['instrument_token']
                            for item in instruments}
        return self.instruments

    def get_historical_data(self, symbol, interval="day", days=None):
        """
        Fetch historical OHLCV data for the given symbol and timeframe.

        Args:
            symbol (str)     : Trading symbol e.g. 'RELIANCE'
            interval (str)   : "day", "15minute", "60minute"
            days (int|None)  : History length. If None, uses sensible defaults
                               based on Zerodha API limits:
                                  daily     -> 200 days
                                  60minute  -> 60 days
                                  15minute  -> 20 days
        Returns:
            pandas.DataFrame : OHLCV dataframe sorted by date
        """

        token = self.instruments.get(symbol)
        if not token:
            print(f"❌ Instrument token not found for {symbol}")
            return None

        # Zerodha historical data limits
        interval_limits = {
            "day":       200,
            "60minute":   60,
            "15minute":   20
        }

        if days is None:
            days = interval_limits.get(interval, 200)
        else:
            # Enforce max allowed by Zerodha
            max_allowed = interval_limits.get(interval, days)
            if days > max_allowed:
                print(f"⚠️ {interval} data is limited to {max_allowed} days by Zerodha API. Adjusting days.")
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
        """Get current market price"""
        token = self.instruments.get(symbol)
        if token:
            quote = self.kite.quote(f"NSE:{symbol}")
            return quote[f"NSE:{symbol}"]["last_price"]
        return None
