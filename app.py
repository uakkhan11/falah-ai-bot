# app.py – Falāh Streamlit Dashboard

import streamlit as st
import subprocess
import os
import time
from datetime import datetime
API_KEY = "ijzeuwuylr3g0kug"

LOG_FILE = "/root/falah-ai-bot/monitor.log"

st.set_page_config(page_title="Falāh Bot Dashboard", layout="wide")

st.title("🟢 Falāh Trading Bot Dashboard")

from token_loader import load_or_create_token

@st.cache_resource
def init_kite():
    return load_or_create_token(
        api_key=API_KEY,
        api_secret="ijzeuwuylr3g0kug",
        interactive=True,
        st=st
    )


# ---------------------------
# Monitor Service Status
# ---------------------------
def get_service_status():
    try:
        output = subprocess.check_output(
            ["systemctl", "is-active", "falah_monitor.service"],
            stderr=subprocess.STDOUT
        )
        return output.decode().strip()
    except subprocess.CalledProcessError:
        return "unknown"

status = get_service_status()
st.info(f"🔄 Monitor Service Status: **{status.upper()}**")

# ---------------------------
# Start/Stop Buttons
# ---------------------------
col1, col2 = st.columns(2)

if col1.button("▶️ Start Monitor"):
    subprocess.run(["systemctl", "start", "falah_monitor.service"])
    st.success("✅ Started the monitor service.")

if col2.button("⏹️ Stop Monitor"):
    subprocess.run(["systemctl", "stop", "falah_monitor.service"])
    st.warning("🛑 Stopped the monitor service.")

# ---------------------------
# Manual Trigger
# ---------------------------
if st.button("🕹️ Run Monitor Now (One Cycle)"):
    with st.spinner("Running monitoring cycle..."):
        os.system("python3 /root/falah-ai-bot/run_monitor.py & sleep 10 && pkill -f run_monitor.py")
    st.success("✅ One monitoring cycle completed.")

# ---------------------------
# Show Logs
# ---------------------------
st.subheader("📄 Recent Logs")

if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    if lines:
        st.text_area(
            "Monitor Log (last 100 lines)",
            "".join(lines[-100:]),
            height=400
        )
    else:
        st.info("No logs yet.")
else:
    st.info("Log file not found.")

# ---------------------------
# Timestamp
# ---------------------------
st.caption(f"Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
