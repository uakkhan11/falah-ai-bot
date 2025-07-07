# debug_missing.py
import json
import pandas as pd
import gspread

# Load secrets
with open("/root/falah-ai-bot/secrets.json") as f:
    secrets = json.load(f)

SPREADSHEET_KEY = secrets["sheets"]["SPREADSHEET_KEY"]

# Load NSE instruments
df = pd.read_csv("/root/falah-ai-bot/all_nse_instruments.csv")

# Load Halal symbols
gc = gspread.service_account(filename="/root/falah-ai-bot/falah-credentials.json")
sheet = gc.open_by_key(SPREADSHEET_KEY)
ws = sheet.worksheet("HalalList")
symbols = [s.strip().upper() for s in ws.col_values(1)[1:] if s.strip()]

nse_symbols = set(df["tradingsymbol"].str.upper())

# Find unmatched
unmatched = sorted(s for s in symbols if s not in nse_symbols)

print(f"❌ {len(unmatched)} Halal symbols not found in NSE instruments:")
print(unmatched)
