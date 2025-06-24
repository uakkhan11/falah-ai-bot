# app.py – Falāh Bot Main UI with Monitor Integration

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
    return client.open_by_key(SHEET_KEY)

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

st.success(f"✅ {len(symbols)} Halal stocks loaded")

if st.checkbox("🔍 Show Halal Symbols"):
    st.write(symbols)

# ---------------------------
# 📈 AI Trade Preview
# ---------------------------

st.title("📈 Falāh AI Trading Bot (Demo UI)")
st.caption("Built with 💡 by Usman on GitHub")

@st.cache_data
def get_live_data(symbols):
    results = []
    for sym in symbols[:10]:
        try:
            ltp_data = kite.ltp(f"NSE:{sym}")
            cmp = ltp_data[f"NSE:{sym}"]["last_price"]
            ai_score = round(random.uniform(60, 95), 2)
            results.append({"Symbol": sym, "CMP": cmp, "AI Score": ai_score})
        except Exception as e:
            st.warning(f"Skipping {sym}: {e}")
    return results

st.info("⏳ Analyzing halal stocks...")
analyzed = get_live_data(symbols)
df = pd.DataFrame(analyzed)

st.subheader("📊 Today's Halal Trade Candidates")
st.dataframe(df, use_container_width=True)

if not df.empty:
    top = df.sort_values(by="AI Score", ascending=False).iloc[0]
    st.success(f"✅ *Top Trade Pick:* `{top['Symbol']}` with AI Score: {top['AI Score']}`)

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
