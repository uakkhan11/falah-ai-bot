# dashboard.py

import streamlit as st
import subprocess
import os
import signal
import json
from kiteconnect import KiteConnect
from credentials import load_secrets

st.set_page_config(page_title="Falāh Trading Bot Dashboard", layout="wide")

st.markdown("# 🟢 Falāh Trading Bot Dashboard")

# --- Monitor Service Status ---

pid_file = "/root/falah-ai-bot/monitor.pid"

def is_monitor_running():
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            pid = int(f.read())
        try:
            os.kill(pid, 0)  # check signal
            return True
        except ProcessLookupError:
            return False
    return False

monitor_running = is_monitor_running()

status_text = "🟢 RUNNING" if monitor_running else "🔴 STOPPED"
status_color = "success" if monitor_running else "error"

st.status = getattr(st, status_color)
st.status(f"Monitor Service Status: **{status_text}**")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Monitor"):
        if monitor_running:
            st.warning("Monitor is already running.")
        else:
            proc = subprocess.Popen(
                ["nohup", "python", "monitor_runner.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(pid_file, "w") as f:
                f.write(str(proc.pid))
            st.success("Monitor started.")
            st.experimental_rerun()

with col2:
    if st.button("🟥 Stop Monitor"):
        if not monitor_running:
            st.warning("Monitor is not running.")
        else:
            with open(pid_file, "r") as f:
                pid = int(f.read())
            os.kill(pid, signal.SIGTERM)
            os.remove(pid_file)
            st.success("Monitor stopped.")
            st.experimental_rerun()

with col3:
    if st.button("🔄 Run Monitor Now (One Cycle)"):
        subprocess.run(["python", "monitor_runner.py", "--once"])
        st.success("One monitoring cycle completed.")

# --- Access Token Management ---
with st.expander("🔑 Access Token Management"):
    st.subheader("Generate New Access Token")

    secrets = load_secrets()
    api_key = secrets["zerodha"]["api_key"]
    api_secret = secrets["zerodha"]["api_secret"]

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    st.markdown(f"[🔗 Click here to login to Zerodha]({login_url})")
    request_token = st.text_input("Paste the request_token here")

    if st.button("Generate Access Token"):
        if not request_token:
            st.error("Please paste your request_token.")
        else:
            try:
                data = kite.generate_session(request_token, api_secret=api_secret)
                access_token = data["access_token"]

                with open("/root/falah-ai-bot/access_token.json", "w") as f:
                    json.dump({"access_token": access_token}, f)

                st.success("✅ Access token generated and saved.")
            except Exception as e:
                st.error(f"Error generating access token: {e}")

st.caption("Falāh Bot © 2025")
