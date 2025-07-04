from kiteconnect import KiteConnect
import json

# Load secrets
with open("/root/falah-ai-bot/secrets.json") as f:
    secrets = json.load(f)

api_key = secrets["zerodha"]["api_key"]

# Load access token
with open("/root/falah-ai-bot/access_token.json") as f:
    access_token = json.load(f)["access_token"]

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Fetch instruments dump
print("Fetching instruments from NSE...")
instruments = kite.instruments("NSE")

# Halal symbols list - 🟢 Replace or extend as needed
halal_symbols = [
    "INFY",
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "HINDUNILVR",
    "BAJFINANCE",
    "KOTAKBANK",
    "LT",
    "MARUTI",
    "AXISBANK",
    "ASIANPAINT",
    "ITC",
    "ULTRACEMCO",
    "WIPRO",
    "TECHM",
    "NESTLEIND",
    "POWERGRID",
    "HCLTECH"
]

token_map = {}
for i in instruments:
    if i["tradingsymbol"] in halal_symbols:
        token_map[i["tradingsymbol"]] = i["instrument_token"]

if not token_map:
    print("⚠️ No tokens matched. Double-check your symbol names.")
else:
    # Save to tokens.json
    with open("/root/falah-ai-bot/tokens.json", "w") as f:
        json.dump(token_map, f, indent=2)

    print(f"✅ tokens.json updated with {len(token_map)} symbols.")
