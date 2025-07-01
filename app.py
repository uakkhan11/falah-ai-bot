# app.py – Falāh Streamlit Dashboard

import streamlit as st
import subprocess
import os
import time
import json
from datetime import datetime
from kiteconnect import KiteConnect

API_KEY = "ijzeuwuylr3g0kug"
API_SECRET = "yy1wd2wn8r0wx4mus00vxllgss03nuqx"
LOG_FILE = "/root/falah-ai-bot/monitor.log"

st.set_page_config(page_title="Falāh Bot Dashboard", layout="wide")
st.title("🟢 Falāh Trading Bot Dashboard")


# ---------------------------
# Token Generation Function
# ---------------------------
def generate_and_save_token(request_token):
    kite = KiteConnect(api_key=API_KEY)
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]

    with open("/root/falah-ai-bot/access_token.json", "w") as f:
        json.dump({"access_token": access_token}, f)

    return access_token


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
# 🔑 Access Token Management
# ---------------------------
st.subheader("🔑 Access Token Management")

with st.expander("Generate New Access Token"):
    st.markdown(
        """
        👉 **Step 1:** Login here and get your `request_token`:  
        [Zerodha Login](https://kite.trade/connect/login?v=3&api_key=ijzeuwuylr3g0kug)
        """
    )
    request_token = st.text_input("Paste Request Token Here")

    if st.button("Generate & Save Access Token"):
        if request_token.strip():
            try:
                token = generate_and_save_token(request_token.strip())
                st.success(f"✅ Access token saved: {token[:4]}... (truncated)")

                # Optionally restart monitor
                subprocess.run(["systemctl", "restart", "falah_monitor.service"])
                st.info("🔄 Monitor service restarted to use new token.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ Please enter a valid request token.")


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
