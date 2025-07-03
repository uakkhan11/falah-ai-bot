# dashboard.py

import streamlit as st
import subprocess
import json
from kiteconnect import KiteConnect
from credentials import load_secrets

st.set_page_config(page_title="Falāh Trading Bot Dashboard", layout="wide")

st.markdown("# 🟢 Falāh Trading Bot Dashboard")

# 💡 Monitor service status (dummy logic for illustration)
service_status = "UNKNOWN"

# --- Monitor Controls ---
st.info(f"Monitor Service Status: **{service_status}**")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Monitor"):
        st.success("Monitor service started (placeholder).")
        # Here you could run: subprocess.Popen(["python", "run_monitor.py"])

with col2:
    if st.button("🟥 Stop Monitor"):
        st.warning("Monitor service stopped (placeholder).")
        # Here you could stop: subprocess.run(["pkill", "-f", "run_monitor.py"])

with col3:
    if st.button("🔄 Run Monitor Now (One Cycle)"):
        st.success("One monitoring cycle completed (placeholder).")
        # You could run: subprocess.run(["python", "run_monitor.py", "--once"])

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

                # Save to access_token.json
                with open("/root/falah-ai-bot/access_token.json", "w") as f:
                    json.dump({"access_token": access_token}, f)

                st.success("✅ Access token generated and saved.")
            except Exception as e:
                st.error(f"Error generating access token: {e}")

st.caption("Falāh Bot © 2025")
