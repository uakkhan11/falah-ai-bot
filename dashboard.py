# dashboard.py

import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

from instrument_loader import save_token_map
from fetch_intraday_data import fetch_intraday_data
from intraday_scanner import scan_intraday_folder
from utils import get_halal_list
from amfi_fetcher import load_large_midcap_symbols

INTRADAY_DIR = "/root/falah-ai-bot/intraday_data/"
SCREENED_FILE = "/root/falah-ai-bot/final_screened.json"

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

st.title("📊 AI Trading Dashboard")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Step 1 — Load symbols (Halal + Large/Mid Cap)
with st.spinner("Loading Halal + L/M Cap symbols..."):
    halal_list = set(get_halal_list("1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"))
    lm_list = set(load_large_midcap_symbols())
    final_symbols = sorted(list(halal_list.intersection(lm_list)))

if not final_symbols:
    st.error("❌ No symbols found in Halal + L/M Cap filter. Check your list.")
    st.stop()

st.success(f"✅ Loaded {len(final_symbols)} filtered symbols.")

# Step 2 — Update token map
with st.spinner("Updating token map..."):
    try:
        save_token_map()
        st.success("✅ Token map updated.")
    except Exception as e:
        st.error(f"❌ Failed to update token map: {e}")
        st.stop()

# Step 3 — Fetch intraday data
with st.spinner("Fetching intraday data..."):
    try:
        fetch_intraday_data(final_symbols)
        st.success("✅ Intraday data fetched.")
    except Exception as e:
        st.error(f"❌ Failed to fetch intraday data: {e}")
        st.stop()

# Step 4 — Run intraday scan
with st.spinner("Scanning intraday data..."):
    try:
        scan_results = scan_intraday_folder(INTRADAY_DIR)
        if scan_results.empty:
            st.warning("⚠️ No symbols matched the scan criteria.")
        else:
            st.success(f"✅ Found {len(scan_results)} matching symbols.")
    except Exception as e:
        st.error(f"❌ Scan failed: {e}")
        st.stop()

# Step 5 — Display results
if not scan_results.empty:
    st.subheader("📌 Scan Results")
    st.dataframe(scan_results)

    # Save scan results for backtesting
    try:
        scan_results.to_csv("latest_intraday_scan.csv", index=False)
        with open(SCREENED_FILE, "w") as f:
            json.dump(scan_results.to_dict(orient="records"), f, indent=2)
        st.success("✅ Scan results saved for backtesting & monitoring.")
    except Exception as e:
        st.error(f"❌ Failed to save scan results: {e}")

    # Place Order buttons
    st.subheader("🛒 Place Orders")
    for _, row in scan_results.iterrows():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        col1.write(row['symbol'])
        col2.write(f"₹{row['close']}")
        col3.write(f"RSI: {row['RSI']}")
        if col4.button(f"Buy {row['symbol']}"):
            st.info(f"Placing order for {row['symbol']}...")
            # TODO: Add order placement logic
else:
    st.info("📭 No trades to display right now.")

st.divider()
st.caption("Powered by AI Trading Bot — Live Intraday Analysis")
