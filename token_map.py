import json
from kiteconnect import KiteConnect

kite = KiteConnect(api_key="YOUR_API_KEY")
kite.set_access_token("YOUR_ACCESS_TOKEN")

# Fetch all instruments from NSE
instruments = kite.instruments(exchange="NSE")

token_map = {}
for instrument in instruments:
    token_map[instrument['tradingsymbol']] = instrument['instrument_token']

with open("/root/falah-ai-bot/token_map.json", "w") as f:
    json.dump(token_map, f, indent=2)

print("✅ token_map.json created with", len(token_map), "symbols")
