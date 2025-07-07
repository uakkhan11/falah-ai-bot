from kiteconnect import KiteConnect
import json
from credentials import load_secrets
from utils import get_halal_list

print("Fetching instruments from NSE...")

# Load secrets
secrets = load_secrets()
api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]
spreadsheet_key = secrets["sheets"]["SPREADSHEET_KEY"]

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Fetch all instruments
instruments = kite.instruments(exchange="NSE")

# Get your Halal list
halal_symbols = set(get_halal_list(spreadsheet_key))

print(f"✅ Loaded {len(halal_symbols)} Halal symbols.")

# Create token mapping
token_map = {}
for i in instruments:
    tradingsymbol = i["tradingsymbol"]
    token = i["instrument_token"]
    if tradingsymbol in halal_symbols:
        token_map[tradingsymbol] = token

print(f"✅ tokens.json updated with {len(token_map)} symbols.")

with open("/root/falah-ai-bot/tokens.json", "w") as f:
    json.dump(token_map, f, indent=2)
