# generate_tokens.py

from kiteconnect import KiteConnect
from utils import load_credentials, get_halal_list
import json

secrets = load_credentials()
kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
kite.set_access_token(secrets["zerodha"]["access_token"])

print("Fetching instruments from NSE...")
instruments = kite.instruments("NSE")

halal_symbols = set(get_halal_list())
token_map = {}

for inst in instruments:
    if inst["tradingsymbol"] in halal_symbols:
        token_map[inst["tradingsymbol"]] = inst["instrument_token"]

print(f"✅ Mapped {len(token_map)} Halal symbols.")
with open("/root/falah-ai-bot/tokens.json", "w") as f:
    json.dump(token_map, f, indent=2)
