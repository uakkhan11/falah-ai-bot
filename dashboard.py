import streamlit as st
import pandas as pd
import subprocess
import os
import signal
import json
from datetime import datetime
from pytz import timezone
from kiteconnect import KiteConnect

from credentials import load_secrets, get_kite, validate_kite
from data_fetch import get_live_ltp
from fetch_historical_batch import fetch_all_historical
from smart_scanner import run_smart_scan
from ws_live_prices import start_all_websockets
from stock_analysis import analyze_stock, get_regime
from bulk_analysis import analyze_multiple_stocks
from telegram_utils import send_telegram

# Helper functions
def compute_trailing_sl(cmp, atr, atr_multiplier=1.5):
    return round(cmp - atr * atr_multiplier, 2)

def log_trade_to_sheet(sheet, timestamp, symbol, quantity, price, action, note):
    sheet.append_row([
        timestamp, symbol, quantity, price, action, note
    ])

def is_market_open():
    india = timezone("Asia/Kolkata")
    now = datetime.now(india)
    return (
        now.weekday() < 5 and
        now.hour >= 9 and
        (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
    )

# Initialize
secrets = load_secrets()
BOT_TOKEN = secrets["telegram"]["bot_token"]
CHAT_ID = secrets["telegram"]["chat_id"]
SPREADSHEET_KEY = secrets["google"]["spreadsheet_key"]

st.set_page_config(page_title="Falāh Bot Dashboard", layout="wide")
st.title("🟢 Falāh Trading Bot Dashboard")

# Monitor Service
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

# Capital Settings
st.sidebar.header("⚙️ Capital & Trade Settings")
total_capital = st.sidebar.number_input("Total Daily Capital (₹)", min_value=1000, value=100000, step=5000)
max_trades = st.sidebar.slider("Max Number of Trades", 1, 10, 5)
dry_run = st.sidebar.checkbox("Dry Run Mode (No Orders)", value=True)

# Auto Scanner
st.subheader("🔍 Auto Scan for New Stocks")
if st.button("Scan Stocks"):
    st.info("Running scanner...")
    df = run_smart_scan()
    if df.empty:
        st.warning("No signals.")
    else:
        st.session_state["scanned_data"] = df

if "scanned_data" in st.session_state:
    df = st.session_state["scanned_data"]

    # Automatically select top N stocks based on Score
    top_n = max_trades
    df_top = df.sort_values(by="Score", ascending=False).head(top_n)
    selected_symbols = df_top["Symbol"].tolist()

    st.success(f"✅ Automatically selected Top {top_n} stocks:")
    st.dataframe(df_top, use_container_width=True)

    if st.button("🚀 Place Orders for Selected"):
        st.info("Placing orders...")
        kite = get_kite()
        if not validate_kite(kite):
            st.error("Invalid token.")
            st.stop()

        df_top["Weight"] = df_top["Score"] / df_top["Score"].sum()

        for _, row in df_top.iterrows():
            sym = row["Symbol"]
            weight = row["Weight"]
            cmp = get_live_ltp(kite, sym)
            allocated_capital = total_capital * weight
            qty = max(1, int(allocated_capital / cmp))
            trailing_sl = compute_trailing_sl(cmp, row.get("ATR", 1))
            target_price = round(cmp + (cmp - trailing_sl) * 3, 2)

            msg = (
                f"🚀 <b>Auto Trade</b>\n"
                f"{sym}\nQty: {qty}\nEntry: ₹{cmp}\nSL: ₹{trailing_sl}\nTarget: ₹{target_price}\nScore: {row['Score']}"
            )

            if dry_run:
                st.success(f"(Dry Run) {msg}")
                send_telegram(BOT_TOKEN, CHAT_ID, f"[DRY RUN]\n{msg}")
            else:
                if not is_market_open():
                    st.warning(f"Market closed for {sym}. Skipping.")
                    continue
                try:
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
                    send_telegram(BOT_TOKEN, CHAT_ID, msg)

                    from gspread import authorize
                    from oauth2client.service_account import ServiceAccountCredentials
                    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                    creds = ServiceAccountCredentials.from_json_keyfile_name("falah-credentials.json", scope)
                    gc = authorize(creds)
                    sheet = gc.open_by_key(SPREADSHEET_KEY).worksheet("TradeLog")
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_trade_to_sheet(sheet, timestamp, sym, qty, cmp, "BUY", f"SL:{trailing_sl} Target:{target_price}")

                except Exception as e:
                    st.error(f"Error placing order for {sym}: {e}")

# Everything else (manual lookup, bulk analysis, bot controls) remains unchanged
