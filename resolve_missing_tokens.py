# resolve_missing_tokens.py

import json
from kiteconnect import KiteConnect
from credentials import load_secrets

# 🔹 Your missing tokens list
missing_tokens = [
    1850625, 3465729, 81153, 492033, 1510401, 1270529,
    341249, 2815745, 2953217, 969473, 424961, 779521
]

# 🔹 Load Zerodha credentials
secrets = load_secrets()
kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
kite.set_access_token(secrets["zerodha"]["access_token"])

print("🔄 Fetching all NSE instruments...")
instruments = kite.instruments(exchange="NSE")
print(f"✅ Loaded {len(instruments)} instruments.")

# 🔹 Build token-to-symbol map
token_to_symbol = {i["instrument_token"]: i["tradingsymbol"] for i in instruments}

print("\n🔍 Resolving missing tokens:\n")
for token in missing_tokens:
    sym = token_to_symbol.get(token)
    if sym:
        print(f"✅ Token {token}: {sym}")
    else:
        print(f"❌ Token {token}: Not found in NSE instruments")
