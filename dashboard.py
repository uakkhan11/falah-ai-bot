# dashboard.py
import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
import subprocess

st.set_page_config(page_title="AI Trading Dashboard", layout="centered")

# Paths
FILTERED_FILE = "/root/falah-ai-bot/final_screened.json"
HISTORICAL_DIR = "/root/falah-ai-bot/historical_data"
INTRADAY_DIR = "/root/falah-ai-bot/intraday_data"

# Utilities
def log_status(msg, status="info"):
    if status == "success":
        st.success(msg)
    elif status == "error":
        st.error(msg)
    elif status == "warning":
        st.warning(msg)
    else:
        st.info(msg)

def run_script(script, args=None):
    cmd = ["python3", script]
    if args:
        cmd += args
    try:
        output = subprocess.check_output(cmd, text=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"❌ Error running {script}: {e.output}"

# Title
st.markdown("## 📊 AI Trading Dashboard")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 1. Enter Token
if st.button("🔑 Enter Zerodha Token"):
    output = run_script("token_entry.py")
    log_status(output, "success")

# 2. Fetch Historical Data
if st.button("📥 Fetch Historical Data"):
    output = run_script("fetch_historical.py")
    log_status(output, "success")

# 3. Fetch Intraday Data
if st.button("⏳ Fetch Intraday Data"):
    output = run_script("fetch_intraday_data.py")
    log_status(output, "success")

# 4. Run Combined Scan
if st.button("🔍 Run Daily + Intraday Scan"):
    # Run daily scan
    run_script("daily_scanner.py")
    # Run intraday scan
    run_script("intraday_scanner.py")

    # Load results
    daily_df = pd.read_csv("daily_screening_results.csv") if os.path.exists("daily_screening_results.csv") else pd.DataFrame()
    intra_df = pd.read_csv("intraday_screening_results.csv") if os.path.exists("intraday_screening_results.csv") else pd.DataFrame()

    if not daily_df.empty or not intra_df.empty:
        st.subheader("📈 Combined Scan Results")
        combined_df = pd.concat([daily_df, intra_df]).drop_duplicates(subset="symbol")
        st.dataframe(combined_df, use_container_width=True)
    else:
        log_status("No matching stocks found", "warning")

# 5. Place Orders
if st.button("💰 Place Orders for All Scanned Stocks"):
    output = run_script("place_orders.py")
    log_status(output, "success")

# 6. Start Monitor
if st.button("📡 Start Monitor"):
    output = run_script("monitor.py")
    log_status(output, "success")

# Notes
st.markdown("---")
st.markdown("✅ **This dashboard always uses your Halal + L/M Cap list.**\n"
            "✅ **Trade reports are sent to Telegram automatically.**\n"
            "✅ **Scans use the latest real-time data before placing orders.**")
