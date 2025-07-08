import streamlit as st
import pandas as pd
import subprocess
import os
import signal
import json
import matplotlib.pyplot as plt
from kiteconnect import KiteConnect

from credentials import load_secrets, get_kite, validate_kite
from data_fetch import fetch_historical_candles, get_live_ltp
from fetch_historical_batch import fetch_all_historical
from smart_scanner import run_smart_scan
from ws_live_prices import start_all_websockets

from stock_analysis import analyze_stock, get_regime
from bulk_analysis import analyze_multiple_stocks
from telegram_utils import send_telegram

st.set_page_config(page_title="Falāh Bot Dashboard", layout="wide")
st.title("🟢 Falāh Trading Bot Dashboard")

# ===== Monitor Service Status =====
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
            st.success("Monitor started.")

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

# ===== Access Token Management =====
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

                with open("/root/falah-ai-bot/secrets.json", "r") as f:
                    secrets_data = json.load(f)
                secrets_data["zerodha"]["access_token"] = access_token
                with open("/root/falah-ai-bot/secrets.json", "w") as f:
                    json.dump(secrets_data, f, indent=2)

                with open("/root/falah-ai-bot/access_token.json", "w") as f:
                    json.dump({"access_token": access_token}, f)

                st.success("✅ Access token saved successfully.")
            except Exception as e:
                st.error(f"Error: {e}")

# ===== Capital Settings =====
st.sidebar.header("⚙️ Capital & Trade Settings")
total_capital = st.sidebar.number_input("Total Daily Capital (₹)", min_value=1000, value=100000, step=5000)
max_trades = st.sidebar.slider("Max Number of Trades", 1, 10, 5)
dry_run = st.sidebar.checkbox("Dry Run Mode (No Orders)", value=True)

# ===== Auto Scanner =====
st.subheader("🔍 Auto Scan for New Stocks")
if st.button("Scan Stocks"):
    st.info("Running scanner...")
    df = run_smart_scan()
    if df.empty:
        st.warning("No signals.")
    else:
        st.session_state["scanned_data"] = df
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
                st.error("Invalid token.")
                st.stop()
            for sym in selected:
                try:
                    cmp = get_live_ltp(kite, sym)
                    qty = int(per_trade_capital / cmp)
                    if dry_run:
                        st.success(f"(Dry Run) Order prepared for {sym} (Qty={qty})")
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

# ===== Manual Stock Lookup =====
st.subheader("🔍 Manual Stock Lookup")
symbol_input = st.text_input("Enter NSE Symbol (e.g., INFY)").strip().upper()
if st.button("Fetch Stock Data"):
    if not symbol_input:
        st.warning("Enter a symbol.")
    else:
        kite = get_kite()
        if not validate_kite(kite):
            st.error("Invalid token.")
            st.stop()
        try:
            result = analyze_stock(kite, symbol_input)
            st.write(f"✅ CMP: ₹{result['cmp']:.2f}")
            st.write(f"ATR(14): {result['atr']:.2f}")
            st.write(f"Trailing SL: ₹{result['trailing_sl']:.2f}")
            st.write(f"ADX: {result['adx']:.2f} ({get_regime(result['adx'])})")
            st.write(f"RSI: {result['rsi']:.2f} ({result['rsi_percentile']*100:.1f}% percentile)")
            st.write(
                f"Bollinger Bands: Upper ₹{result['bb_upper']:.2f}, Mid ₹{result['bb_mid']:.2f}, Lower ₹{result['bb_lower']:.2f}"
            )
            st.write(f"Relative Strength: {result['rel_strength']:.2f}")
            st.write(f"Risk per share: ₹{result['risk']:.2f}, Reward per share: ₹{result['reward']:.2f}")
            st.write(f"Backtest Win Rate: {result['backtest_winrate']:.1f}%")
            st.write(f"🚦 Recommendation: **{result['recommendation']}**")
            st.write(f"AI Exit Score: {result['ai_score']}")
            st.write("Reasons:", result["reasons"])
            st.dataframe(result["history"].tail(10))

            fig, ax = plt.subplots()
            ax.plot(result["history"]["Date"], result["history"]["RSI"], label="RSI")
            ax.axhline(70, color="red", linestyle="--")
            ax.axhline(30, color="green", linestyle="--")
            ax.set_title("RSI Over Time")
            ax.legend()
            st.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.plot(result["history"]["Date"], result["history"]["Close"], label="Close")
            ax2.plot(result["history"]["Date"], result["history"]["Supertrend"], label="Supertrend")
            ax2.set_title("Price vs Supertrend")
            ax2.legend()
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Error fetching data: {e}")

# ===== Bulk Analysis =====
st.subheader("📊 Bulk Stock Analysis")
symbols_input = st.text_area(
    "Enter NSE symbols separated by commas (e.g., INFY,TCS,HDFCBANK):"
).strip().upper()

if st.button("Analyze Stocks"):
    if not symbols_input:
        st.warning("Enter at least one symbol.")
    else:
        symbols_list = [s.strip() for s in symbols_input.split(",")]
        kite = get_kite()
        if not validate_kite(kite):
            st.error("Invalid token.")
            st.stop()
        st.info("Analyzing...")
        results = analyze_multiple_stocks(kite, symbols_list)

        rows = []
        for r in results:
            if "error" in r:
                rows.append({"Symbol": r["symbol"], "Error": r["error"]})
            else:
                rows.append({
                    "Symbol": r["symbol"],
                    "CMP": r["cmp"],
                    "ADX": r["adx"],
                    "RSI": r["rsi"],
                    "RelStrength": r["rel_strength"],
                    "AI_Score": r["ai_score"],
                    "Recommendation": r["recommendation"]
                })
        df = pd.DataFrame(rows)
        st.dataframe(df)

# ===== Bot Controls =====
st.subheader("⚙️ Bot Controls")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Fetch Historical Data"):
        fetch_all_historical()
        st.success("Historical data fetched.")

with col2:
    if st.button("▶️ Start Live WebSockets"):
        start_all_websockets()
        st.success("WebSockets started.")

with col3:
    if st.button("🛑 Stop Live WebSockets"):
        st.warning("Stop functionality not implemented yet.")

if os.path.exists("/root/falah-ai-bot/last_fetch.txt"):
    with open("/root/falah-ai-bot/last_fetch.txt") as f:
        ts = f.read()
    st.info(f"Last historical fetch: {ts}")
