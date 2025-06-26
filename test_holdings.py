from kiteconnect import KiteConnect
from utils import load_credentials, get_cnc_holdings

secrets = load_credentials()
kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
kite.set_access_token(secrets["zerodha"]["access_token"])

holdings = get_cnc_holdings(kite)
print(f"✅ CNC Holdings: {len(holdings)}")
for h in holdings:
    print(h["tradingsymbol"], h["quantity"], h["average_price"])
