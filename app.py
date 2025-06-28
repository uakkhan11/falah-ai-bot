# app.py – Falāh Bot Main UI with Monitor Integration + Smart Scanner + WebSocket Live Data

import streamlit as st
import pandas as pd
import time
import random
import subprocess
import os
from kiteconnect import KiteConnect, KiteTicker
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
    try:
        profile = kite.profile()
        st.success(f"🧑‍💼 Logged in as: {profile['user_name']}")
    except Exception as e:
        st.error(f"❌ Failed to fetch profile: {e}")
    return kite

@st.cache_resource
def load_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_JSON, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_KEY)
    required_sheets = ["HalalList", "LivePositions", "MonitoredStocks"]
    existing_titles = [ws.title for ws in sheet.worksheets()]
    for title in required_sheets:
        if title not in existing_titles:
            sheet.add_worksheet(title=title, rows=1000, cols=20)
    return sheet

def get_halal_symbols(sheet):
    worksheet = sheet.worksheet("HalalList")
    all_symbols = worksheet.col_values(1)
    return [s.strip() for s in all_symbols[1:] if s.strip()]

def is_monitor_running():
    try:
        status = subprocess.check_output(["systemctl", "is-active", "monitor.service"]).decode().strip()
        return status == "active"
    except:
        return False

# --------------------
# 📡 WebSocket Live Data
# --------------------
live_ltps = {}

def start_websocket(tokens):
    kws = KiteTicker(API_KEY, ACCESS_TOKEN)

    def on_connect(ws, response):
        st.write("✅ WebSocket connected.")
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_ticks(ws, ticks):
        for tick in ticks:
            instrument_token = tick["instrument_token"]
            ltp = tick.get("last_price")
            if ltp is not None:
                live_ltps[instrument_token] = ltp

    def on_error(ws, code, reason):
        st.warning(f"⚠️ WebSocket error: {code} {reason}")

    def on_close(ws, code, reason):
        st.warning("🔌 WebSocket closed.")

    kws.on_connect = on_connect
    kws.on_ticks = on_ticks
    kws.on_error = on_error
    kws.on_close = on_close
    kws.connect(threaded=True)

# ---------------------------
# 🖥️ Monitor Controls
# ---------------------------
st.markdown("## 🔍 Falāh Live Monitor")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚀 Start Monitoring"):
        subprocess.run(["sudo", "systemctl", "start", "monitor.service"])
        st.success("✅ Monitor started")
with col2:
    if st.button("🛑 Stop Monitoring"):
        subprocess.run(["sudo", "systemctl", "stop", "monitor.service"])
        st.warning("🛑 Monitor stopped")
with col3:
    if st.button("🧪 Run Monitor Once Now"):
        subprocess.run(["python3", "/root/falah-ai-bot/monitor.py"])
        st.success("✅ Monitor executed manually")

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

try:
    symbols = get_halal_symbols(sheet)
    st.success(f"✅ {len(symbols)} Halal stocks loaded")
except Exception as e:
    st.error(f"❌ Failed to load symbols: {e}")
    symbols = []

if st.checkbox("🔍 Show All Halal Symbols"):
    st.write(symbols)

if st.button("⚡ Run Smart Scanner"):
    with st.spinner("Running multi-timeframe scanner..."):
        subprocess.run(["python3", "/root/falah-ai-bot/smart_scanner.py"])
    st.success("✅ Scanner finished.")

if os.path.exists("/root/falah-ai-bot/scan_results.csv"):
    df_scan = pd.read_csv("/root/falah-ai-bot/scan_results.csv")
    st.subheader("📊 Last Scan Results")
    st.dataframe(df_scan)
else:
    st.info("📭 No scan results yet.")

# ---------------------------
# 📈 AI Trade Preview & Fund Management
# ---------------------------
st.title("📈 Falāh AI Trading Bot")

st.sidebar.header("⚙️ Fund Management")
enable_dummy = st.sidebar.checkbox("🧪 Enable Dummy Mode", value=False)
total_capital = st.sidebar.number_input("💰 Total Capital", 1000, 1000000, 100000)
max_trades = st.sidebar.number_input("📈 Max Trades", 1, 20, 5)
min_ai_score = st.sidebar.slider("🎯 Min AI Score", 0, 100, 70)

# 🚀 Start WebSocket subscription
if not enable_dummy:
    # Get tokens of first 10 symbols for demonstration
    try:
        instruments = kite.ltp([f"NSE:{s}" for s in symbols[:10]])
        tokens = [info["instrument_token"] for info in instruments.values()]
        start_websocket(tokens)
        st.success(f"✅ WebSocket started for tokens: {tokens}")
    except Exception as e:
        st.warning(f"⚠️ Could not start WebSocket: {e}")

# 🔄 Live Data Fetch
@st.cache_data
def get_live_data(symbols):
    results = []
    for sym in symbols:
        try:
            if enable_dummy:
                cmp = round(random.uniform(200, 1500), 2)
            else:
                # Get instrument token for this symbol
                inst = kite.ltp(f"NSE:{sym}")
                token = inst[f"NSE:{sym}"]["instrument_token"]
                cmp = live_ltps.get(token)
                if cmp is None:
                    cmp = inst[f"NSE:{sym}"]["last_price"]
            ai_score = round(random.uniform(60, 95), 2)
            results.append({"Symbol": sym, "CMP": cmp, "AI Score": ai_score})
        except Exception as e:
            st.warning(f"❌ Skipping {sym}: {e}")
    return results

st.info("⏳ Analyzing...")
analyzed = get_live_data(symbols)
df = pd.DataFrame(analyzed)

if df.empty or "AI Score" not in df.columns:
    st.error("❌ No valid data.")
    st.stop()
else:
    st.success("✅ Data fetched.")
    st.write(df.head())

candidates = df[df["AI Score"] >= min_ai_score].sort_values(by="AI Score", ascending=False).head(max_trades)

if not candidates.empty:
    total_score = candidates["AI Score"].sum()
    candidates["Weight"] = candidates["AI Score"] / total_score
    candidates["Allocation"] = (candidates["Weight"] * total_capital).round(2)
    candidates["Est. Qty"] = (candidates["Allocation"] / candidates["CMP"]).astype(int)
    st.dataframe(candidates)

    if st.button("🛒 Execute Trades Now"):
        for _, row in candidates.iterrows():
            if row["Est. Qty"] <= 0:
                continue
            try:
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
                st.success(f"✅ Order placed for {row['Symbol']}")
            except Exception as e:
                st.warning(f"⚠️ Order failed for {row['Symbol']}: {e}")
else:
    st.warning("⚠️ No candidates met the threshold.")

# ---------------------------
# 📦 Live Position Viewer
# ---------------------------
st.markdown("---")
st.subheader("📦 Live Positions")
try:
    ws = sheet.worksheet("LivePositions")
    records = ws.get_all_records()
    if records:
        df_live = pd.DataFrame(records)
        st.dataframe(df_live)
    else:
        st.info("📭 No positions.")
except Exception as e:
    st.warning(f"⚠️ Could not load positions: {e}")

# ---------------------------
# 🧾 Monitored Stocks Viewer
# ---------------------------
st.subheader("🧾 Monitored CNC Holdings")
try:
    ws_monitor = sheet.worksheet("MonitoredStocks")
    records = ws_monitor.get_all_records()
    if records:
        df_monitor = pd.DataFrame(records)
        st.dataframe(df_monitor)
    else:
        st.info("📭 No monitored stocks.")
except Exception as e:
    st.warning(f"⚠️ Could not load monitored stocks: {e}")

st.caption("Built with 💡 by Usman")

st.markdown("---")
st.subheader("🟢 Live WebSocket LTP Monitor")

if st.button("🔄 Refresh Live LTPs"):
    if live_ltps:
        df_ltp = pd.DataFrame([
            {"Token": k, "LTP": v} for k, v in list(live_ltps.items())[:20]
        ])
        st.dataframe(df_ltp)
    else:
        st.warning("⚠️ No live LTP data received yet.")

