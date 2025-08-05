import os, json, subprocess, signal, base64
from datetime import datetime
from pytz import timezone

import streamlit as st
import pandas as pd
import psutil
import joblib
from kiteconnect import KiteConnect

# Local imports
from credentials import load_secrets, get_kite, validate_kite
from data_fetch import fetch_all_historical
from fetch_intraday_data import fetch_intraday_data
from daily_scanner import scan_daily_folder
from intraday_scanner import scan_intraday_folder
from stock_analysis import analyze_stock, get_regime
from bulk_analysis import analyze_multiple_stocks
from ws_live_prices import start_all_websockets
from telegram_utils import send_telegram
from sheets import log_trade_to_sheet
from live_price_reader import get_symbol_price_map

# =======================
# 🌙 Streamlit Setup
# =======================
st.set_page_config(page_title="Falāh Bot Dashboard", layout="wide", page_icon="🌙")

# Background Image
def set_bg(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: 40%;
        background-position: top right;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: #fdf9f2;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("icon-512.png")

# =======================
# Title
# =======================
st.markdown("""
<div style='text-align: center; margin-top: -30px;'>
<h1>🟢 <b>Falāh Trading Bot</b></h1>
<h4 style='color: #6b7280;'>Ethical • Intelligent • Profitable</h4>
</div>
""", unsafe_allow_html=True)

# =======================
# Load Model
# =======================
@st.cache_resource
def load_model():
    return joblib.load("/root/falah-ai-bot/model.pkl")

model = load_model()

# Utility functions
def compute_trailing_sl(cmp, atr, atr_multiplier=1.5):
    return round(cmp - atr * atr_multiplier, 2)

def calculate_quantity(capital, risk_pct, entry, sl):
    risk_amt = capital * risk_pct
    per_share_risk = entry - sl
    if per_share_risk <= 0:
        raise ValueError("SL must be below entry.")
    return max(int(risk_amt / per_share_risk), 1)

def is_market_open():
    now = datetime.now(timezone("Asia/Kolkata"))
    return now.weekday() < 5 and (9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30))

# =======================
# Load Secrets
# =======================
secrets = load_secrets()
BOT_TOKEN = secrets["telegram"]["bot_token"]
CHAT_ID = secrets["telegram"]["chat_id"]

# =======================
# Access Token Management
# =======================
with st.expander("🔑 Access Token Management"):
    st.subheader("Generate New Access Token")
    api_key = secrets["zerodha"]["api_key"]
    api_secret = secrets["zerodha"]["api_secret"]
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    st.markdown(f"[🔗 Click here to login to Zerodha]({login_url})")

    request_token = st.text_input("Paste request_token here")
    if st.button("Generate Access Token"):
        if not request_token:
            st.error("Please paste the request_token.")
        else:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                access_token = data["access_token"]
                secrets_path = "/root/falah-ai-bot/secrets.json"
                if os.path.exists(secrets_path):
                    with open(secrets_path, "r") as f:
                        secrets_data = json.load(f)
                    secrets_data["zerodha"]["access_token"] = access_token
                    with open(secrets_path, "w") as f:
                        json.dump(secrets_data, f, indent=2)
                st.success("✅ Access token generated and saved successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# =======================
# Sidebar Settings
# =======================
st.sidebar.header("⚙️ Trade Settings")
capital = st.sidebar.number_input("Daily Capital (₹)", 1000, 10_00_000, 100_000, 5000)
max_trades = st.sidebar.slider("Max Trades", 1, 10, 5)
dry_run = st.sidebar.toggle("Dry Run Mode", value=True)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
min_conf = st.sidebar.slider("Min AI Confidence", 0.1, 1.0, 0.25, 0.05)

# =======================
# Live LTP Monitor
# =======================
st.subheader("🔴 Live LTP Monitor")
try:
    prices = get_symbol_price_map()
    if prices:
        df = pd.DataFrame([{"Symbol": sym, "LTP": ltp} for sym, ltp in prices.items()])
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("📴 No live prices available.")
except Exception as e:
    st.error(f"❌ Failed to fetch live prices: {e}")

# =======================
# 📅 Daily Scanner
# =======================
st.subheader("📅 Daily Scanner")
if st.button("Run Daily Scan"):
    results = scan_daily_folder("historical_data/")
    if not results.empty:
        st.success(f"✅ {len(results)} stock(s) passed.")
        st.dataframe(results)
        st.download_button("⬇ Download", results.to_csv(index=False), file_name="daily_results.csv")
    else:
        st.warning("⚠️ No stocks passed daily scan.")

# =======================
# ⏱️ Intraday Scanner
# =======================
st.subheader("⏱️ Intraday Scanner")
if st.button("Run Intraday Scan"):
    results = scan_intraday_folder("intraday_data/")
    if not results.empty:
        st.success(f"✅ {len(results)} stock(s) passed.")
        st.dataframe(results)
        st.download_button("⬇ Download", results.to_csv(index=False), file_name="intraday_results.csv")
    else:
        st.warning("⚠️ No stocks passed intraday scan.")

# =======================
# 🔍 Manual Stock Lookup
# =======================
st.subheader("🔍 Manual Stock Lookup")
symbol_input = st.text_input("Enter NSE Symbol").strip().upper()
if st.button("Fetch Stock Data"):
    kite = get_kite()
    if not validate_kite(kite):
        st.error("Invalid token.")
    else:
        try:
            result = analyze_stock(kite, symbol_input)
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")

# =======================
# 📊 Bulk Analysis
# =======================
st.subheader("📊 Bulk Stock Analysis")
symbols_input = st.text_area("Enter symbols separated by commas:").strip().upper()
if st.button("Analyze Stocks"):
    if symbols_input:
        symbols_list = [s.strip() for s in symbols_input.split(",")]
        kite = get_kite()
        if validate_kite(kite):
            results = analyze_multiple_stocks(kite, symbols_list)
            st.dataframe(pd.DataFrame(results))
        else:
            st.error("Invalid token.")
    else:
        st.warning("Enter at least one symbol.")

# =======================
# ⚙️ Bot Controls
# =======================
st.subheader("⚙️ Bot Controls")
c1, c2, c3 = st.columns(3)
if c1.button("📥 Fetch Historical"): fetch_all_historical(); st.success("✅ Done")
if c2.button("▶️ Start Websockets"): start_all_websockets(); st.success("✅ Started")
if c3.button("🛑 Stop Websockets"): st.info("Stop not implemented.")

# =======================
# 🕐 Intraday Data Fetcher
# =======================
st.subheader("🕐 Intraday Data Fetcher")
interval = st.selectbox("Timeframe", ["15minute", "60minute"])
days = st.slider("Past days", 1, 10, 5)
if st.button("📥 Fetch Intraday Data"):
    try:
        fetch_intraday_data([], interval=interval, days=days)
        st.success("✅ Intraday data fetched.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
