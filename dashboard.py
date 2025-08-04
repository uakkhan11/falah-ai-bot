# dashboard.py
import os
import json
import subprocess
import signal
from datetime import datetime
from pytz import timezone

import pandas as pd
import streamlit as st
from kiteconnect import KiteConnect
import joblib
import psutil

from credentials import load_secrets, get_kite, validate_kite
from fetch_intraday_data import fetch_intraday_data
from fetch_historical_batch import fetch_all_historical
from intraday_scanner import scan_intraday_folder
from daily_scanner import scan_daily_folder
from telegram_utils import send_telegram
from live_price_reader import get_symbol_price_map
from sheets import log_trade_to_sheet

# ───────────────────────────────
# STREAMLIT PAGE CONFIG
# ───────────────────────────────
st.set_page_config(page_title="Falāh Bot", layout="wide", page_icon="🌙")

# ───────────────────────────────
# LOAD MODEL
# ───────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("/root/falah-ai-bot/model.pkl")

model = load_model()

# ───────────────────────────────
# COMMON FUNCTIONS
# ───────────────────────────────
def is_market_open():
    now = datetime.now(timezone("Asia/Kolkata"))
    return now.weekday() < 5 and (9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30))

def calculate_quantity(capital, risk_pct, entry, sl):
    risk_amt = capital * risk_pct
    per_share_risk = entry - sl
    if per_share_risk <= 0:
        raise ValueError("SL must be below entry.")
    return max(int(risk_amt / per_share_risk), 1)

def monitor_status():
    pid_file = "/root/falah-ai-bot/monitor.pid"
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            if psutil.pid_exists(pid):
                p = psutil.Process(pid)
                if "python" in p.name() and "monitor_runner.py" in ' '.join(p.cmdline()):
                    return True
        except:
            pass
    return False

# ───────────────────────────────
# SIDEBAR SETTINGS
# ───────────────────────────────
st.sidebar.header("⚙️ Trade Settings")
capital = st.sidebar.number_input("Daily Capital (₹)", 1000, 10_00_000, 100_000, 5000)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
max_trades = st.sidebar.slider("Max Trades", 1, 20, 5)
min_conf = st.sidebar.slider("Min AI Confidence", 0.1, 1.0, 0.25, 0.05)

# ───────────────────────────────
# MONITOR CONTROLS
# ───────────────────────────────
st.subheader("🖥️ Monitor Controls")
status = "🟢 RUNNING" if monitor_status() else "🔴 STOPPED"
st.info(f"Monitor Status: **{status}**")

c1, c2, c3 = st.columns(3)
if c1.button("▶️ Start Monitor") and not monitor_status():
    proc = subprocess.Popen(
        ["nohup", "python3", "monitor_runner.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    with open("/root/falah-ai-bot/monitor.pid", "w") as f:
        f.write(str(proc.pid))
    st.success("✅ Monitor started.")
    st.rerun()

if c2.button("🟥 Stop Monitor") and monitor_status():
    try:
        with open("/root/falah-ai-bot/monitor.pid", "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        os.remove("/root/falah-ai-bot/monitor.pid")
        st.success("✅ Monitor stopped.")
    except Exception as e:
        st.error(f"❌ Failed to stop monitor: {e}")
    st.rerun()

if c3.button("🔄 Run Once"):
    subprocess.run(["python3", "monitor_runner.py", "--once"])
    st.success("✅ Cycle complete.")

# ───────────────────────────────
# SCAN & ORDER SECTION
# ───────────────────────────────
st.subheader("📊 Full Scan & Trade")

if st.button("🚀 Run Full Scan"):
    st.info("⏳ Fetching latest data...")
    try:
        # Step 1: Fetch latest data
        fetch_all_historical()
        fetch_intraday_data(symbols=[], interval="15minute", days=5)  # Empty list = all configured

        # Step 2: Run scans
        daily_df = scan_daily_folder("/root/falah-ai-bot/historical_data")
        intraday_df = scan_intraday_folder("/root/falah-ai-bot/intraday_data")

        # Step 3: Merge results
        scanned_df = pd.concat([daily_df, intraday_df]).drop_duplicates(subset=["symbol"])
        scanned_df = scanned_df.sort_values("ai_score", ascending=False).head(max_trades)

        if scanned_df.empty:
            st.warning("⚠️ No stocks matched the filters.")
        else:
            st.success(f"✅ Found {len(scanned_df)} candidates.")
            st.dataframe(scanned_df)

            # Step 4: Get latest LTP
            st.info("📡 Getting latest LTPs before placing orders...")
            price_map = get_symbol_price_map()

            kite = get_kite()
            if not validate_kite(kite):
                st.error("⚠️ Invalid access token.")
                st.stop()

            # Step 5: Place orders
            for _, row in scanned_df.iterrows():
                sym = row["symbol"]
                cmp = price_map.get(sym, row["close"])
                if pd.isna(cmp):
                    st.warning(f"⚠️ Skipping {sym}, no price found.")
                    continue

                if row["ai_score"] < min_conf:
                    st.warning(f"❌ Skipped {sym} (Conf: {row['ai_score']:.2f})")
                    continue

                sl = round(cmp * 0.985, 2)
                try:
                    qty = calculate_quantity(capital, risk_pct, cmp, sl)
                except ValueError as ve:
                    st.warning(f"⚠️ {sym}: {ve}")
                    continue

                msg = f"🚀 <b>{sym}</b>\nQty: {qty}\nEntry: ₹{cmp}\nSL: ₹{sl}\nConf: {row['ai_score']:.2f}"
                
                if is_market_open():
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
                        st.success(f"✅ Order placed for {sym} | Qty: {qty} | Price: {cmp}")
                        send_telegram(load_secrets()["telegram"]["bot_token"], load_secrets()["telegram"]["chat_id"], msg)
                        log_trade_to_sheet(
                            symbol=sym, qty=qty, price=cmp, rsi=row["RSI"], atr=None,
                            adx=None, ai_score=row["ai_score"], action="BUY",
                            exit_reason="", pnl="", outcome=""
                        )
                    except Exception as e:
                        st.error(f"❌ Order failed for {sym}: {e}")
                else:
                    st.warning("⏸️ Market is closed.")

    except Exception as e:
        st.error(f"❌ Scan failed: {e}")
