# app.py – Falāh Bot Main UI with Monitor Integration (Improved with Smart Fund Management + Auto Buy)

import streamlit as st
import pandas as pd
import time
import random
import subprocess
import os
from kiteconnect import KiteConnect
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import toml

# 🔐 Load credentials
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)

API_KEY = secrets["zerodha"]["api_key"]
API_SECRET = secrets["zerodha"]["api_secret"]
ACCESS_TOKEN = secrets["zerodha"]["access_token"]
CREDS_JSON = "falah-credentials.json"
SHEET_KEY = secrets.get("global", {}).get("google_sheet_key", "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")

st.set_page_config(page_title="Falāh Bot UI", layout="wide")

# --------------------
# Kite + Sheet Init
# --------------------

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
    sheet = client.open_by_key(SHEET_KEY)
    # Ensure required worksheet exists
    required_sheets = ["HalalList", "LivePositions"]
    existing_titles = [ws.title for ws in sheet.worksheets()]
    for title in required_sheets:
        if title not in existing_titles:
            sheet.add_worksheet(title=title, rows=1000, cols=20)
    return sheet

def get_halal_symbols(sheet):
    worksheet = sheet.worksheet("HalalList")
    return worksheet.col_values(1)[1:]

# ---------------------------
# 🖥️ Monitor Controls
# ---------------------------

def is_monitor_running():
    try:
        status = subprocess.check_output(["systemctl", "is-active", "monitor.service"]).decode().strip()
        return status == "active"
    except:
        return False

st.markdown("## 🔍 Falāh Live Monitor")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Start Monitoring"):
        subprocess.run(["sudo", "systemctl", "start", "monitor.service"])
        st.success("✅ Monitor started")
with col2:
    if st.button("🛑 Stop Monitoring"):
        subprocess.run(["sudo", "systemctl", "stop", "monitor.service"])
        st.warning("🛑 Monitor stopped")

if is_monitor_running():
    st.info("📡 *Monitor is running* and tracking CNC holdings.")
else:
    st.error("⚠️ Monitor is not active.")

# ---------------------------
# 📜 Halal Stock Scanner
# ---------------------------

st.title("📜 Falāh Halal Stock Scanner")

kite = init_kite()
sheet = load_sheet()
symbols = get_halal_symbols(sheet)
st.write("📋 Loaded symbols:", symbols[:10])

st.success(f"✅ {len(symbols)} Halal stocks loaded")

if st.checkbox("🔍 Show Halal Symbols"):
    st.write(symbols)

# ---------------------------
# 📈 AI Trade Preview & Fund Management
# ---------------------------

st.title("📈 Falāh AI Trading Bot")
st.caption("Built with 💡 by Usman")

# Fund and risk controls
st.sidebar.header("⚙️ Fund Management")
enable_dummy = st.sidebar.checkbox("🧪 Enable Dummy Data Mode", value=False)
total_capital = st.sidebar.number_input("💰 Total Capital (₹)", min_value=1000, value=100000, step=1000)
max_trades = st.sidebar.number_input("📈 Max Trades Per Day", min_value=1, value=5, step=1)
min_ai_score = st.sidebar.slider("🎯 Min AI Score to Consider", 0, 100, 70)

@st.cache_data
def get_live_data(symbols):
    results = []
    for sym in symbols[:10]:  # limit for debug
        try:
            ltp_data = kite.ltp(f"NSE:{sym}")
            cmp = ltp_data[f"NSE:{sym}"]["last_price"]
            ai_score = round(random.uniform(60, 95), 2)
            results.append({"Symbol": sym, "CMP": cmp, "AI Score": ai_score})
        except Exception as e:
            st.warning(f"❌ Skipping {sym}: {e}")
    return results

st.info("⏳ Analyzing halal stocks...")
analyzed = get_live_data(symbols)
st.write("✅ Raw data from get_live_data():", analyzed)
@st.cache_data
def get_live_data(symbols):
    results = []
    for sym in symbols[:10]:  # limit for debug
        try:
            if enable_dummy:
                cmp = round(random.uniform(200, 1500), 2)
            else:
                ltp_data = kite.ltp(f"NSE:{sym}")
                cmp = ltp_data[f"NSE:{sym}"]["last_price"]

            ai_score = round(random.uniform(60, 95), 2)
            results.append({"Symbol": sym, "CMP": cmp, "AI Score": ai_score})
        except Exception as e:
            st.warning(f"❌ Skipping {sym}: {e}")
    return results
    
df = pd.DataFrame(analyzed)
if df.empty or "AI Score" not in df.columns:
    st.error("❌ No valid stock data fetched. Please verify your Zerodha API credentials or enable dummy mode.")
    st.stop()
else:
    df = pd.DataFrame()
    st.warning("⚠️ No stock data available. Check if Zerodha access token is valid or API is rate-limited.")
    st.write("🧾 Raw DataFrame:", df)

st.subheader("📊 Filtered Trade Candidates")

if not df.empty and "AI Score" in df.columns:
    candidates = df.sort_values(by="AI Score", ascending=False).head(max_trades)
else:
    candidates = pd.DataFrame()
    st.warning("⚠️ No valid trade candidates available due to missing or invalid data.")

if not candidates.empty:
    total_score = candidates["AI Score"].sum()
    candidates["Weight"] = candidates["AI Score"] / total_score
    candidates["Allocation"] = (candidates["Weight"] * total_capital).round(2)
    candidates["Est. Qty"] = (candidates["Allocation"] / candidates["CMP"]).astype(int)
    st.dataframe(candidates, use_container_width=True)
    top = candidates.iloc[0]
    st.success(f"✅ *Top Pick:* `{top['Symbol']}` – Score: {top['AI Score']}, Capital: ₹{top['Allocation']}")

    if st.button("🛒 Execute Trades Now"):
        for _, row in candidates.iterrows():
            try:
                if row["Est. Qty"] <= 0:
                    continue
                kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NSE,
                    tradingsymbol=row["Symbol"],
                    transaction_type=kite.TRANSACTION_TYPE_BUY,
                    quantity=int(row["Est. Qty"]),
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_CNC
                )
                sheet.worksheet("LivePositions").append_row([
                    row["Symbol"],
                    int(row["Est. Qty"]),
                    float(row["CMP"]),
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])
                st.success(f"✅ Order placed for {row['Symbol']} – Qty: {int(row['Est. Qty'])}")
            except Exception as e:
                st.warning(f"⚠️ Failed to place order for {row['Symbol']}: {e}")
else:
    st.warning("⚠️ No trade candidates met the minimum AI score threshold.")

# ---------------------------
# 📦 CNC Position Viewer
# ---------------------------

st.markdown("---")
st.subheader("📦 Live Position Tracker")
try:
    ws = sheet.worksheet("LivePositions")
    records = ws.get_all_records()
    if records:
        df_live = pd.DataFrame(records)
        st.dataframe(df_live, use_container_width=True)
    else:
        st.info("📭 No live positions currently being tracked.")
except Exception as e:
    st.warning(f"⚠️ Failed to fetch positions: {e}")

st.caption("This dashboard auto-monitors Halal positions for SL/Target and includes Telegram alerts + smart exit logic.")
