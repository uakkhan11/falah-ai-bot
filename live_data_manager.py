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
    
    def get_historical_data(self, symbol, days=200):
        """Get historical data for indicator calculation"""
        token = self.instruments.get(symbol)
        if not token:
            return None
            
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        data = self.kite.historical_data(
            instrument_token=token,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
            interval="day"
        )
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        return df
    
    def get_current_price(self, symbol):
        """Get current market price"""
        token = self.instruments.get(symbol)
        if token:
            quote = self.kite.quote(f"NSE:{symbol}")
            return quote[f"NSE:{symbol}"]["last_price"]
        return None
