from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import pytz
import pandas as pd

IST = pytz.timezone("Asia/Kolkata")

API_KEY = "ijzeuwuylr3g0kug"
ACCESS_TOKEN = "AmrLL5hewKt70qzmrJCisiJEmlSP9b4M"

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

symbol = "NCC"

instrument = kite.ltp(f"NSE:{symbol}")
instrument_token = instrument[f"NSE:{symbol}"]["instrument_token"]

to_date = datetime.now(IST)
from_date = to_date - timedelta(days=60)

candles = kite.historical_data(
    instrument_token,
    from_date,
    to_date,
    "day"
)

df = pd.DataFrame(candles)
print(f"Rows fetched: {len(df)}")
print(df.head(20))
