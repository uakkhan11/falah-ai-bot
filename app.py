import streamlit as st
import pandas as pd
import random
from kiteconnect import KiteConnect
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -----------------------------
# 🔐 Load credentials
# -----------------------------
API_KEY = st.secrets["zerodha"]["api_key"]
API_SECRET = st.secrets["zerodha"]["api_secret"]
ACCESS_TOKEN = st.secrets["zerodha"]["access_token"]
CREDS_JSON = "falah-credentials.json"
SHEET_KEY = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"

st.set_page_config(page_title="Falāh Bot", layout="wide")
st.title("📈 Falāh AI Trading Bot")
st.caption("Built with 💡 by Usman — Live CMP, Halal picks, and CNC tracker")

# -----------------------------
# 🔧 Init Kite & Google Sheet
# -----------------------------
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

# -----------------------------
# 🚀 Load data
# -----------------------------
kite = init_kite()
sheet = load_sheet()
symbols = get_halal_symbols(sheet)
st.success(f"✅ {len(symbols)} Halal stocks loaded")

if st.checkbox("🔍 Show Halal Symbols"):
    st.write(symbols)

# -----------------------------
# 🤖 AI Stock Scoring Engine
# -----------------------------
@st.cache_data
def get_live_data(symbols):
    results = []
    for sym in symbols[:10]:  # Limit to 10 for preview
        try:
            ltp_data = kite.ltp(f"NSE:{sym}")
            cmp = ltp_data[f"NSE:{sym}"]["last_price"]
            ai_score = round(random.uniform(60, 95), 2)  # Replace with real AI logic
            results.append({"Symbol": sym, "CMP": cmp, "AI Score": ai_score})
        except Exception as e:
            st.warning(f"⚠️ Skipping {sym}: {e}")
    return results

st.info("⏳ Analyzing Halal stocks...")
analyzed = get_live_data(symbols)
df = pd.DataFrame(analyzed)

if not df.empty:
    st.subheader("📊 Today's Halal Trade Candidates")
    st.dataframe(df, use_container_width=True)
    top = df.sort_values(by="AI Score", ascending=False).iloc[0]
    st.success(f"✅ *Top Trade Pick:* `{top['Symbol']}` with AI Score: {top['AI Score']}")
else:
    st.warning("No valid stock data to display.")

# -----------------------------
# 📦 Live CNC Position Tracker
# -----------------------------
st.markdown("---")
st.subheader("📦 Live Position Tracker")

try:
    positions = kite.positions()["net"]
    live_status = []

    for pos in positions:
        if pos["product"] != "CNC" or pos["quantity"] == 0:
            continue
        symbol = pos["tradingsymbol"]
        buy_price = pos["average_price"]
        qty = pos["quantity"]
        cmp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]

        change_pct = round((cmp - buy_price) / buy_price * 100, 2)

        if change_pct <= -3:
            status = "🔻 Stoploss Hit"
        elif change_pct >= 5:
            status = "🎯 Target Hit"
        else:
            status = "🔄 Holding"

        live_status.append({
            "Symbol": symbol,
            "Qty": qty,
            "Buy Price": buy_price,
            "CMP": cmp,
            "Change %": change_pct,
            "Status": status
        })

    if live_status:
        st.dataframe(pd.DataFrame(live_status))
    else:
        st.info("No CNC positions found.")

except Exception as e:
    st.error(f"⚠️ Failed to fetch positions: {e}")

# -----------------------------
# 📤 Optional: Add Sell Trigger, Sheet or Telegram Integration
# -----------------------------
st.markdown("---")
st.caption("This dashboard auto-monitors Halal positions for SL/Target and will soon include Telegram alerts + automated exit logic.")
