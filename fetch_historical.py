from kiteconnect import KiteConnect
import pandas as pd
import os
import toml
import time

# 🔐 Load credentials
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)

API_KEY = secrets["zerodha"]["api_key"]
ACCESS_TOKEN = secrets["zerodha"]["access_token"]

# Initialize Kite
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ✅ Make sure this folder exists
DATA_FOLDER = "/root/falah-ai-bot/data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# ✅ Define date range
from datetime import datetime, timedelta

to_date = datetime.today()
from_date = to_date - timedelta(days=365)  # past 1 year

# ✅ Load Halal symbols from your Google Sheet
import gspread
from oauth2client.service_account import ServiceAccountCredentials

CREDS_JSON = "falah-credentials.json"
SHEET_KEY = secrets.get("global", {}).get("google_sheet_key", "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_JSON, scope)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_KEY)
symbols = sheet.worksheet("HalalList").col_values(1)[1:]

print(f"Loaded {len(symbols)} symbols from HalalList.")

# ✅ Fetch instrument tokens
tokens = {}
with open("/root/falah-ai-bot/tokens.json", "r") as f:
    tokens = json.load(f)
else:
     print(f"⚠️ Skipping {s}: No LTP data available.")

# ✅ Download and save historical data
for symbol in symbols:
    try:
        token = tokens[symbol]
        print(f"Fetching {symbol} ({token})...")
        data = kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval="day",
            continuous=False,
            oi=False
        )
        df = pd.DataFrame(data)
        file_path = os.path.join(DATA_FOLDER, f"{symbol}.csv")
        df.to_csv(file_path, index=False)
        print(f"✅ Saved: {file_path}")
        time.sleep(0.5)  # avoid rate limit
    except Exception as e:
        print(f"❌ Failed for {symbol}: {e}")
