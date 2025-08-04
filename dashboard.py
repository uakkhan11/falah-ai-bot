# dashboard.py
import os, json, subprocess, signal, base64, traceback
from datetime import datetime
from pytz import timezone

import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
import joblib
import psutil

from credentials import load_secrets, get_kite, validate_kite
from fetch_historical_batch import fetch_all_historical
from fetch_intraday_data import fetch_intraday_data
from intraday_scanner import run_intraday_scan
from smart_scanner import run_smart_scan
from live_price_reader import get_symbol_price_map
from telegram_utils import send_telegram
from sheets import log_trade_to_sheet

# ----------------------------------------------------
# Streamlit Config
# ----------------------------------------------------
st.set_page_config(page_title="Falāh Bot", layout="wide", page_icon="🌙")

# Background styling
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
        background-color: #fdf9f2;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("icon-512.png")

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.markdown("""
<div style='text-align: center; margin-top: -30px;'>
<h1>🟢 <b>Falāh Trading Bot</b></h1>
<h4 style='color: #6b7280;'>Ethical • Intelligent • Profitable</h4>
</div>
""", unsafe_allow_html=True)

# Load secrets
secrets = load_secrets()
BOT_TOKEN = secrets["telegram"]["bot_token"]
CHAT_ID = secrets["telegram"]["chat_id"]

# Cached model load
@st.cache_resource
def load_model():
    return joblib.load("/root/falah-ai-bot/model.pkl")

model = load_model()

# Quantity calculation
def calculate_quantity(capital, risk_pct, entry, sl):
    risk_amt = capital * risk_pct
    per_share_risk = entry - sl
    if per_share_risk <= 0:
        raise ValueError("SL must be below entry.")
    return max(int(risk_amt / per_share_risk), 1)

# Market open check
def is_market_open():
    now = datetime.now(timezone("Asia/Kolkata"))
    return now.weekday() < 5 and (9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30))

# ----------------------------------------------------
# Sidebar - Trade Settings
# ----------------------------------------------------
st.sidebar.header("⚙️ Trade Settings")
capital = st.sidebar.number_input("Daily Capital (₹)", 1000, 10_00_000, 100_000, 5000)
max_trades = st.sidebar.slider("Max Trades", 1, 10, 5)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
min_conf = st.sidebar.slider("Min AI Confidence", 0.1, 1.0, 0.25, 0.05)

# ----------------------------------------------------
# Section 1 - Data Fetching
# ----------------------------------------------------
st.subheader("📥 Data Fetching")

c1, c2 = st.columns(2)
if c1.button("📥 Fetch Historical Data"):
    with st.spinner("Fetching historical data..."):
        try:
            fetch_all_historical()
            st.success("✅ Historical data fetched.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

if c2.button("📥 Fetch Intraday Data"):
    with st.spinner("Fetching intraday data..."):
        try:
            fetch_intraday_data(interval="15minute", days=5)
            st.success("✅ Intraday data fetched.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ----------------------------------------------------
# Section 2 - Daily Scan
# ----------------------------------------------------
st.subheader("📊 Daily Scanner")

if st.button("🔍 Run Daily Scan"):
    try:
        scanned_df, debug_stats = run_smart_scan(include_today=True)  # include today’s candle
        if not scanned_df.empty:
            st.success(f"✅ {len(scanned_df)} candidates found.")
            st.dataframe(scanned_df)
            st.session_state["daily_results"] = scanned_df
        else:
            st.warning("⚠️ No candidates found.")
    except Exception as e:
        st.error(f"❌ Scan failed: {e}")
        st.text(traceback.format_exc())

# ----------------------------------------------------
# Section 3 - Intraday Scan
# ----------------------------------------------------
st.subheader("📈 Intraday Scanner")

if st.button("🔍 Run Intraday Scan"):
    try:
        intraday_df, intraday_logs = run_intraday_scan(include_today=True)
        if not intraday_df.empty:
            st.success(f"✅ {len(intraday_df)} intraday candidates found.")
            st.dataframe(intraday_df)
            st.session_state["intraday_results"] = intraday_df
        else:
            st.warning("⚠️ No intraday candidates found.")
    except Exception as e:
        st.error(f"❌ Intraday scan failed: {e}")
        st.text(traceback.format_exc())

# ----------------------------------------------------
# Section 4 - Place Orders
# ----------------------------------------------------
st.subheader("🚀 Place Orders")

def place_orders_from_df(df):
    kite = get_kite()
    if not validate_kite(kite):
        st.error("⚠️ Invalid access token.")
        return

    results_log = []
    prices = get_symbol_price_map(df["symbol"].tolist())

    for _, row in df.iterrows():
        sym = row["symbol"]
        cmp = prices.get(sym, row.get("ltp", 0))
        confidence = row.get("Score", 0)

        if confidence < min_conf:
            st.warning(f"❌ Skipped {sym} (Conf: {confidence:.2f})")
            continue

        sl = round(cmp * 0.985, 2)
        try:
            qty = calculate_quantity(capital, risk_pct, cmp, sl)
        except ValueError as ve:
            st.warning(f"⚠️ {sym}: {ve}")
            continue

        try:
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=sym,
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=qty,
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_CNC
            )
            st.success(f"✅ {sym} order placed @ ₹{cmp} | Qty: {qty}")
            send_telegram(BOT_TOKEN, CHAT_ID, f"🚀 BUY {sym}\nQty: {qty}\nEntry: ₹{cmp}\nSL: ₹{sl}\nConf: {confidence:.2f}")

            # Save to executed trades log for backtesting
            trade_log = {
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                "price": cmp,
                "qty": qty,
                "sl": sl,
                "confidence": confidence
            }
            results_log.append(trade_log)

        except Exception as e:
            st.error(f"❌ {sym} order failed: {e}")

    if results_log:
        trades_file = "executed_trades.csv"
        if os.path.exists(trades_file):
            existing = pd.read_csv(trades_file)
            updated = pd.concat([existing, pd.DataFrame(results_log)], ignore_index=True)
        else:
            updated = pd.DataFrame(results_log)
        updated.to_csv(trades_file, index=False)

if st.button("📦 Place Daily Orders"):
    if "daily_results" in st.session_state:
        place_orders_from_df(st.session_state["daily_results"])
    else:
        st.warning("⚠️ Run Daily Scan first.")

if st.button("📦 Place Intraday Orders"):
    if "intraday_results" in st.session_state:
        place_orders_from_df(st.session_state["intraday_results"])
    else:
        st.warning("⚠️ Run Intraday Scan first.")
