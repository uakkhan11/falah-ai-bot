import json
from kiteconnect import KiteConnect
from utils import load_credentials

# Load secrets.toml credentials
from utils import load_credentials
secrets = load_credentials()
creds = secrets["zerodha"]

# Load access token from JSON
try:
    with open("/root/falah-ai-bot/access_token.json", "r") as f:
        access_token_data = json.load(f)
    access_token = access_token_data["access_token"]
    print(f"✅ Access token loaded: {access_token[:4]}... (truncated)")
except Exception as e:
    print(f"❌ Failed to load access token: {e}")
    exit(1)
    
# Initialize Kite
kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(access_token)

# Read Halal symbols
with open("/root/falah-ai-bot/halal_symbols.txt", "r") as f:
    symbols = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Loaded {len(symbols)} symbols")

# Get instrument dump
print("Fetching instrument dump...")
instruments = kite.instruments("NSE")

token_map = {}
missing = []

for sym in symbols:
    match = next((i for i in instruments if i["tradingsymbol"] == sym), None)
    if match:
        token_map[sym] = match["instrument_token"]
    else:
        missing.append(sym)

# Save tokens.json
with open("/root/falah-ai-bot/tokens.json", "w") as f:
    json.dump(token_map, f, indent=2)

print(f"✅ Saved tokens.json with {len(token_map)} tokens")

if missing:
    print("⚠️ These symbols were not found:")
    for s in missing:
        print(f"- {s}")
