import streamlit as st
import pandas as pd
import subprocess
import os
import signal
import json
from datetime import datetime
from pytz import timezone
from kiteconnect import KiteConnect
import joblib
import base64

from credentials import load_secrets, get_kite, validate_kite
from data_fetch import get_live_ltp
from fetch_historical_batch import fetch_all_historical
from smart_scanner import run_smart_scan
from ws_live_prices import start_all_websockets
from stock_analysis import analyze_stock, get_regime
from bulk_analysis import analyze_multiple_stocks
from telegram_utils import send_telegram
from sheets import log_trade_to_sheet

# ── Page Setup with Logo and Theme ─────────────────────────────

def set_bg(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: 40%;
        background-position: top right;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: #fdf9f2;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg("icon-512.png")

st.set_page_config(page_title="Falāh Bot Dashboard", layout="wide", page_icon="🌙")

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Georgia', serif;
        color: #064e3b;
    }
    .stButton > button {
        background-color: #f59e0b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #d97706;
    }
    .stSidebar, .css-1d391kg, .css-6qob1r {
        background-color: #fefce8 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-top: -30px;'>
        <h1>🟢 <b>Falāh Trading Bot</b></h1>
        <h4 style='color: #6b7280;'>Ethical • Intelligent • Profitable</h4>
    </div>
""", unsafe_allow_html=True)

def inject_pwa():
    st.markdown("""
        <!-- Manifest -->
        <link rel="manifest" href="/static/manifest.json">
        <meta name="theme-color" content="#064e3b">

        <!-- iOS -->
        <link rel="apple-touch-icon" href="/icon-512.png">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">

        <!-- Register Service Worker -->
        <script>
        if ("serviceWorker" in navigator) {
            navigator.serviceWorker.register("/static/sw.js")
                .then(reg => console.log("✅ Service worker registered."))
                .catch(err => console.error("Service worker failed:", err));
        }
        </script>
    """, unsafe_allow_html=True)

inject_pwa()

# Load ML model

def get_trade_probability(rsi, atr, ema10, ema21, volume_change):
    model = joblib.load("model.pkl")
    features = pd.DataFrame([{
        "RSI": rsi,
        "EMA10": ema10,
        "EMA21": ema21,
        "ATR": atr,
        "VolumeChange": volume_change
    }])
    prob = model.predict_proba(features)[0][1]
    return prob

# Helper functions

def compute_trailing_sl(cmp, atr, atr_multiplier=1.5):
    return round(cmp - atr * atr_multiplier, 2)

def calculate_risk_based_quantity(capital: float, risk_per_trade_pct: float, entry_price: float, stoploss_price: float) -> int:
    risk_per_trade_amount = capital * risk_per_trade_pct
    risk_per_share = entry_price - stoploss_price
    if risk_per_share <= 0:
        raise ValueError("Stoploss must be below entry price.")
    qty = max(int(risk_per_trade_amount / risk_per_share), 1)
    return qty

def is_market_open():
    india = timezone("Asia/Kolkata")
    now = datetime.now(india)
    return now.weekday() < 5 and now.hour >= 9 and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))

# Secrets
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

# Access Token Management
with st.expander("🔑 Access Token Management"):
    st.subheader("Generate New Access Token")
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

                st.success("✅ Access token saved.")
            except Exception as e:
                st.error(f"Error: {e}")

# Capital Settings
st.sidebar.header("⚙️ Capital & Trade Settings")
total_capital = st.sidebar.number_input("Total Daily Capital (₹)", min_value=1000, value=100000, step=5000)
max_trades = st.sidebar.slider("Max Number of Trades", 1, 10, 5)
dry_run = st.sidebar.checkbox("Dry Run Mode (No Orders)", value=True)

risk_per_trade_pct = st.sidebar.slider(
    "Risk per Trade (%)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
) / 100

min_confidence = st.sidebar.slider(
    "Minimum Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

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
            cmp = get_live_ltp(kite, sym)
            rsi = row["RSI"]
            atr = row["ATR"]
            adx = row["ADX"]
            ema10 = row["EMA10"]
            ema21 = row["EMA21"]
            volume_change = row["VolumeChange"]


            trailing_sl = compute_trailing_sl(cmp, atr)
            target_price = round(cmp + (cmp - trailing_sl) * 3, 2)

            confidence = get_trade_probability(rsi, atr, ema10, ema21, volume_change)
            ai_score = round(confidence * 5, 2)
            st.write(f"Predicted success probability for {sym}: {confidence:.2f} (AI Score: {ai_score})")

            if confidence < min_confidence:
                st.warning(f"Skipping {sym} due to low confidence ({confidence:.2f} < {min_confidence}).")
                continue

            try:
                qty = calculate_risk_based_quantity(
                    capital=total_capital,
                    risk_per_trade_pct=risk_per_trade_pct,
                    entry_price=cmp,
                    stoploss_price=trailing_sl
                )
            except Exception as e:
                st.error(f"Error calculating quantity: {e}")
                continue

            if confidence >= 0.8:
                qty = int(qty * 1.3)
            elif confidence >= 0.7:
                qty = int(qty * 1.1)
            else:
                qty = int(qty * 0.9)

            msg = (
                f"🚀 <b>Auto Trade</b>\n"
                f"{sym}\nQty: {qty}\nEntry: ₹{cmp}\nSL: ₹{trailing_sl}\nTarget: ₹{target_price}\n"
                f"Confidence: {confidence:.2f}"
            )

            if dry_run:
                st.success(f"(Dry Run) {msg}")
                send_telegram(BOT_TOKEN, CHAT_ID, f"[DRY RUN]\n{msg}")
            else:
                if not is_market_open():
                    st.warning("Market closed. Skipping.")
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

                    log_trade_to_sheet(
                        sheet,
                        timestamp,
                        sym,
                        qty,
                        cmp,
                        "",
                        rsi,
                        atr,
                        adx,
                        ai_score,
                        "BUY",
                        "",
                        "",
                        ""
                    )

                except Exception as e:
                    st.error(f"Error placing order: {e}")

# Manual Stock Lookup
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
            trailing_sl = compute_trailing_sl(result['cmp'], result['atr'])
            target_price = round(result['cmp'] + (result['cmp'] - trailing_sl) * 3, 2)
            st.write(f"Trailing SL: ₹{trailing_sl}")
            st.write(f"Target Price (1:3 R/R): ₹{target_price}")
            st.write(f"ADX: {result['adx']:.2f} ({get_regime(result['adx'])})")
            st.write(f"RSI: {result['rsi']:.2f} ({result['rsi_percentile']*100:.1f}% percentile)")
            st.write(f"Relative Strength: {result['rel_strength']:.2f}")
            st.write(f"AI Exit Score: {result['ai_score']}")
            st.write(f"Recommendation: **{result['recommendation']}**")
            st.dataframe(result["history"].tail(10))

        except Exception as e:
            st.error(f"Error fetching data: {e}")

# Bulk Analysis
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

# Bot Controls
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
