import streamlit as st
import pandas as pd
import subprocess
import os
import signal
import json
from kiteconnect import KiteConnect
from credentials import load_secrets, get_kite, validate_kite
from data_fetch import get_cnc_holdings, get_live_ltp, fetch_historical_candles
from ai_engine import calculate_ai_exit_score
from ta.volatility import AverageTrueRange
from smart_scanner import run_smart_scan
from fetch_historical_batch import fetch_all_historical
from ws_live_prices import start_all_websockets

st.set_page_config(page_title="Falāh Bot Dashboard", layout="wide")

st.title("🟢 Falāh Trading Bot Dashboard")

# Show monitor status message if any
if "monitor_status" in st.session_state:
    st.success(f"Monitor {st.session_state['monitor_status']}.")
    del st.session_state["monitor_status"]

# ======= Monitor Service Status =======
pid_file = "/root/falah-ai-bot/monitor.pid"

def is_monitor_running():
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            pid = int(f.read())
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    return False

monitor_running = is_monitor_running()
status_text = "🟢 RUNNING" if monitor_running else "🔴 STOPPED"
st.info(f"Monitor Service Status: **{status_text}**")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Monitor"):
        if monitor_running:
            st.warning("Already running.")
        else:
            proc = subprocess.Popen(
                ["nohup", "python3", "monitor_runner.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(pid_file, "w") as f:
                f.write(str(proc.pid))
            st.session_state["monitor_status"] = "started"

with col2:
    if st.button("🟥 Stop Monitor"):
        if not monitor_running:
            st.warning("Not running.")
        else:
            with open(pid_file, "r") as f:
                pid = int(f.read())
            os.kill(pid, signal.SIGTERM)
            os.remove(pid_file)
            st.success("Monitor stopped.")
            st.rerun()

with col3:
    if st.button("🔄 Run Monitor Once"):
        subprocess.run(["python3", "monitor_runner.py", "--once"])
        st.success("Monitor cycle complete.")

# ======= Access Token Management =======
with st.expander("🔑 Access Token Management"):
    st.subheader("Generate New Access Token")
    secrets = load_secrets()
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

                # Save to secrets.json
                with open("/root/falah-ai-bot/secrets.json", "r") as f:
                    secrets_data = json.load(f)
                secrets_data["zerodha"]["access_token"] = access_token
                with open("/root/falah-ai-bot/secrets.json", "w") as f:
                    json.dump(secrets_data, f, indent=2)

                # ✅ Also save to access_token.json (needed by get_kite)
                with open("/root/falah-ai-bot/access_token.json", "w") as f:
                    json.dump({"access_token": access_token}, f)

                st.success("✅ Access token saved to both secrets.json and access_token.json.")
            except Exception as e:
                st.error(f"Error: {e}")

# ======= Capital Allocation =======
st.sidebar.header("⚙️ Capital & Trade Settings")

total_capital = st.sidebar.number_input("Total Daily Capital (₹)", min_value=1000, value=100000, step=5000)
max_trades = st.sidebar.slider("Max Number of Trades", 1, 10, 5)
dry_run = st.sidebar.checkbox("Dry Run Mode (No Orders)", value=True)

# ======= Scanner Module =======
st.subheader("🔍 Auto Scan for New Stocks")

if st.button("Scan Stocks"):
    st.info("🔄 Running scanner...")
    df = run_smart_scan()
    if df.empty:
        st.warning("No signals.")
    else:
        st.dataframe(df)

if "scanned_data" in st.session_state:
    df = st.session_state["scanned_data"]
    st.dataframe(df, use_container_width=True)

    selected = st.multiselect("Select stocks to BUY", options=df["Symbol"].tolist())

    if st.button("🚀 Place Orders for Selected"):
        if not selected:
            st.warning("No stocks selected.")
        else:
            st.info("Placing orders...")
            per_trade_capital = total_capital / max_trades

            kite = get_kite()
            if not validate_kite(kite):
                st.error("Invalid token. Please regenerate.")
                st.stop()

            for sym in selected:
                try:
                    cmp = get_live_ltp(kite, sym)
                    qty = int(per_trade_capital / cmp)
                    st.write(f"Placing order for {sym}: Qty={qty}, Price={cmp}")

                    if dry_run:
                        st.success(f"(Dry Run) Order prepared for {sym}")
                    else:
                        kite.place_order(
                            variety=kite.VARIETY_REGULAR,
                            exchange=kite.EXCHANGE_NSE,
                            tradingsymbol=sym,
                            transaction_type=kite.TRANSACTION_TYPE_BUY,
                            quantity=qty,
                            order_type=kite.ORDER_TYPE_MARKET,
                            product=kite.PRODUCT_CNC
                        )
                        st.success(f"✅ Order placed for {sym}")
                except Exception as e:
                    st.error(f"Error placing order for {sym}: {e}")

# ======= Manual Search =======
st.subheader("🔍 Manual Stock Lookup")

symbol_input = st.text_input("Enter NSE Symbol (e.g., INFY)")

if st.button("Fetch Stock Data"):
    if not symbol_input:
        st.warning("Enter a symbol.")
    else:
        kite = get_kite()
        if not validate_kite(kite):
            st.error("Invalid token.")
            st.stop()

        try:
            ltp_data = kite.ltp(f"NSE:{symbol_input}")
            cmp = ltp_data[f"NSE:{symbol_input}"]["last_price"]
            instrument_token = ltp_data[f"NSE:{symbol_input}"]["instrument_token"]

            hist = fetch_historical_candles(
                kite,
                instrument_token=instrument_token,
                interval="day",
                days=30
            )
            df = pd.DataFrame(hist)
            df.columns = [col.capitalize() for col in df.columns]

            st.write(f"✅ Current Market Price: ₹{cmp}")
            st.dataframe(df.tail(10))

            atr = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range().iloc[-1]
            trailing_sl = round(cmp - atr * 1.5, 2)
            st.write(f"ATR(14): {atr:.2f}")
            st.write(f"Recommended ATR-based Trailing SL: ₹{trailing_sl}")

            ai_score, reasons = calculate_ai_exit_score(df, trailing_sl, cmp)
            st.write(f"AI Exit Score: {ai_score}")
            st.write("Reasons:", reasons)
        except Exception as e:
            st.error(f"Error fetching data: {e}")

st.subheader("⚙️ Bot Controls")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Fetch Historical Data"):
        st.info("Fetching historical data...")
        fetch_all_historical()
        st.success("Done fetching.")

with col2:
    if st.button("▶️ Start Live WebSockets"):
        start_all_websockets()
        st.success("WebSockets started.")

with col3:
    if st.button("🛑 Stop Live WebSockets"):
        # This is optional. You can kill them by removing JSON files.
        # Or you can write PID files in ws_live_prices.py and kill them here.
        st.warning("Stop functionality not yet implemented. Use server for now.")

if os.path.exists("/root/falah-ai-bot/last_fetch.txt"):
    with open("/root/falah-ai-bot/last_fetch.txt") as f:
        ts = f.read()
    st.info(f"Last historical fetch: {ts}")

