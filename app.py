import streamlit as st
import pandas as pd
import time
import random
import streamlit as st
from kiteconnect import KiteConnect
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# 🔐 Load credentials (replace with your actual values or use secrets)
API_KEY = st.secrets["zerodha"]["api_key"]
API_SECRET = st.secrets["zerodha"]["api_secret"]
ACCESS_TOKEN = st.secrets["zerodha"]["access_token"]
CREDS_JSON = "falah-credentials.json"
SHEET_KEY = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"


@st.cache_resource
def init_kite():
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite

@st.cache_resource
def load_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_JSON, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_KEY)

def get_halal_symbols(sheet):
    worksheet = sheet.worksheet("HalalList")
    return worksheet.col_values(1)[1:]  # Skip header

st.title("📜 Falāh Halal Stock Scanner")

kite = init_kite()
sheet = load_sheet()
symbols = get_halal_symbols(sheet)

st.success(f"✅ {len(symbols)} Halal stocks loaded")

if st.checkbox("🔍 Show Halal Symbols"):
    st.write(symbols)

# -------------------------------
# Falāh Trading Bot – UI Preview
# -------------------------------

st.set_page_config(page_title="Falāh Bot UI", layout="wide")

st.title("📈 Falāh AI Trading Bot (Demo UI)")
st.caption("Built with 💡 by Usman on GitHub")

# Dummy Halal stocks
sample_stocks = [
    {"Symbol": "TCS", "CMP": 3835.2, "AI Score": 82.5},
    {"Symbol": "INFY", "CMP": 1453.8, "AI Score": 77.4},
    {"Symbol": "LTIM", "CMP": 5450.0, "AI Score": 73.2},
    {"Symbol": "HCLTECH", "CMP": 1289.4, "AI Score": 79.8},
    {"Symbol": "WIPRO", "CMP": 432.1, "AI Score": 68.9}
]

# Simulate AI updates
for stock in sample_stocks:
    stock["AI Score"] += random.uniform(-2.5, 2.5)
    stock["AI Score"] = round(min(100, max(0, stock["AI Score"])), 2)

# Convert to DataFrame
df = pd.DataFrame(sample_stocks)

# Show Table
st.subheader("📊 Today's Halal Trade Candidates")
st.dataframe(df, use_container_width=True)

# Show Top Pick
top = df.sort_values(by="AI Score", ascending=False).iloc[0]
st.success(f"✅ *Top Trade Pick:* `{top['Symbol']}` with AI Score: {top['AI Score']}")

# Footer
st.markdown("---")
st.caption("This is a preview of Falāh Bot's frontend — live trades & AI engine coming soon.")
