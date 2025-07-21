# generate_tokens_from_all.py
import pandas as pd
import json
from kiteconnect import KiteConnect
import gspread

# Load secrets
def load_secrets():
    with open("/root/falah-ai-bot/secrets.json") as f:
        return json.load(f)

secrets = load_secrets()

# Zerodha credentials
api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]

# Google Sheet info
SPREADSHEET_KEY = secrets["google"]["SPREADSHEET_KEY"]

# Init KiteConnect
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

print("🔄 Fetching all NSE instruments from Zerodha...")
instruments = kite.instruments("NSE")
df = pd.DataFrame(instruments)
print(f"✅ Loaded {len(df)} instruments.")

# Save them if you want
df.to_csv("/root/falah-ai-bot/all_nse_instruments.csv", index=False)

# Fetch Halal symbols from Google Sheets
print("🔄 Loading Halal symbols from Google Sheets...")
gc = gspread.service_account(filename="/root/falah-ai-bot/falah-credentials.json")
sheet = gc.open_by_key(SPREADSHEET_KEY)
ws = sheet.worksheet("HalalList")
symbols = ws.col_values(1)[1:]  # skip header
halal_symbols = set(s.strip().upper() for s in symbols if s.strip())

print(f"✅ Loaded {len(halal_symbols)} Halal symbols from sheet.")

# Filter instruments matching Halal symbols
filtered = df[df["tradingsymbol"].str.upper().isin(halal_symbols)]
print(f"✅ Matched {len(filtered)} symbols in Zerodha instruments.")

# Build mapping
token_map = dict(zip(filtered["tradingsymbol"], filtered["instrument_token"]))

# Save tokens.json
with open("/root/falah-ai-bot/tokens.json", "w") as f:
    json.dump(token_map, f, indent=2)

print(f"✅ tokens.json updated with {len(token_map)} symbols.")
