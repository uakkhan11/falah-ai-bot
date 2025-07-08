from credentials import get_kite, validate_kite
import pandas as pd

kite = get_kite()
if not validate_kite(kite):
    print("Invalid Kite token.")
    exit()

instrument_token = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["instrument_token"]

hist = kite.historical_data(
    instrument_token=instrument_token,
    from_date="2023-01-01",
    to_date="2023-12-31",
    interval="day"
)
df = pd.DataFrame(hist)
df.to_csv("historical_data/NIFTY.csv", index=False)
print("✅ Nifty historical data saved.")
