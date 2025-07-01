# app.py – Falāh Bot Mobile Dashboard

import streamlit as st
import threading
import json
import time
import toml
from kiteconnect import KiteConnect, KiteTicker
from monitor_core import monitor_once
from ws_live_prices import live_prices, start_websocket

st.set_page_config(page_title="Falāh Bot", layout="wide")

# Load credentials
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)
API_KEY = secrets["zerodha"]["api_key"]

# Kite init
@st.cache_resource
def init_kite():
    kite = KiteConnect(api_key=API_KEY)
    with open("/root/falah-ai-bot/access_token.json") as f:
        token = json.load(f)["access_token"]
    kite.set_access_token(token)
    return kite

kite = init_kite()

# Load tokens
with open("/root/falah-ai-bot/tokens.json") as f:
    token_map = json.load(f)

# Live LTP
if "live_prices" not in st.session_state:
    st.session_state["live_prices"] = {}

# Monitor thread flag
if "monitor_running" not in st.session_state:
    st.session_state["monitor_running"] = False

# Logs
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def log(message):
    st.session_state.logs.append(message)
    st.session_state.logs = st.session_state.logs[-50:]

# Monitoring loop
def monitoring_loop():
    log("✅ Monitoring started.")
    while st.session_state.monitor_running:
        monitor_once(kite, token_map, log, live_prices)
        time.sleep(900)

# UI
st.title("📈 Falāh Trading Bot")

col1, col2 = st.columns(2)
with col1:
    if st.session_state.monitor_running:
        if st.button("🛑 Stop Monitoring"):
            st.session_state.monitor_running = False
            log("🛑 Monitoring stopped.")
    else:
        if st.button("▶️ Start Monitoring"):
            st.session_state.monitor_running = True
            threading.Thread(target=monitoring_loop, daemon=True).start()

with col2:
    if st.button("🔄 Refresh Access Token"):
        st.warning("Please run get_token.py manually for now.")

st.subheader("🪵 Logs")
st.text("\n".join(st.session_state.logs))
