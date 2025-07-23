import os, json, subprocess, signal, base64
from datetime import datetime
from pytz import timezone

import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
import joblib
import psutil

from credentials import load_secrets, get_kite, validate_kite
from data_fetch import get_live_ltp
from fetch_historical_batch import fetch_all_historical
from smart_scanner import run_smart_scan
from stock_analysis import analyze_stock, get_regime
from bulk_analysis import analyze_multiple_stocks
from ws_live_prices import start_all_websockets
from telegram_utils import send_telegram
from sheets import log_trade_to_sheet

# ✅ Streamlit UI Setup
st.set_page_config(page_title="Falāh Bot Dashboard", layout="wide", page_icon="🌙")

def set_bg(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
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
    """, unsafe_allow_html=True)

set_bg("icon-512.png")

st.markdown("""
<div style='text-align: center; margin-top: -30px;'>
<h1>🟢 <b>Falāh Trading Bot</b></h1>
<h4 style='color: #6b7280;'>Ethical • Intelligent • Profitable</h4>
</div>
""", unsafe_allow_html=True)

# ✅ PWA injection (Optional)
def inject_pwa():
    st.markdown("""
    <link rel="manifest" href="/static/manifest.json">
    <meta name="theme-color" content="#064e3b">
    <link rel="apple-touch-icon" href="/icon-512.png">
    <script>
    if ("serviceWorker" in navigator) {
        navigator.serviceWorker.register("/static/sw.js")
            .then(reg => console.log("✅ SW registered"))
            .catch(err => console.error("SW failed:", err));
    }
    </script>
    """, unsafe_allow_html=True)

inject_pwa()

# ✅ Cached Model Load
@st.cache_resource
def load_model():
    return joblib.load("/root/falah-ai-bot/model.pkl")

model = load_model()

def get_trade_probability(rsi, atr, ema10, ema21, volchg, adx):
    import pandas as pd
    features_df = pd.DataFrame([{
        'RSI': rsi,
        'ATR': atr,
        'ADX': adx,
        'EMA10': ema10,
        'EMA21': ema21,
        'VolumeChange': volchg
    }])
    return model.predict_proba(features_df)[0][1]

def compute_trailing_sl(cmp, atr, atr_multiplier=1.5): return round(cmp - atr * atr_multiplier, 2)

def calculate_quantity(capital, risk_pct, entry, sl):
    risk_amt = capital * risk_pct
    per_share_risk = entry - sl
    if per_share_risk <= 0: raise ValueError("SL must be below entry.")
    return max(int(risk_amt / per_share_risk), 1)

def is_market_open():
    now = datetime.now(timezone("Asia/Kolkata"))
    return now.weekday() < 5 and (9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30))

# ✅ Secrets
secrets = load_secrets()
BOT_TOKEN, CHAT_ID, SPREADSHEET_KEY = secrets["telegram"]["bot_token"], secrets["telegram"]["chat_id"], secrets["google"]["spreadsheet_key"]

# ✅ ✅ ✅ Access Token Management
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

                # Update secrets.json
                secrets_path = "/root/falah-ai-bot/secrets.json"
                if os.path.exists(secrets_path):
                    with open(secrets_path, "r") as f:
                        secrets_data = json.load(f)
                    secrets_data["zerodha"]["access_token"] = access_token
                    with open(secrets_path, "w") as f:
                        json.dump(secrets_data, f, indent=2)

                # Save to access_token.json
                with open("/root/falah-ai-bot/access_token.json", "w") as f:
                    json.dump({"access_token": access_token}, f)

                st.success("✅ Access token generated and saved successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ✅ Sidebar Settings
st.sidebar.header("⚙️ Capital & Trade Settings")
capital = st.sidebar.number_input("Daily Capital (₹)", 1000, 10_00_000, 100_000, 5000)
max_trades = st.sidebar.slider("Max Trades", 1, 10, 5)
dry_run = st.sidebar.toggle("Dry Run Mode", value=True)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
min_conf = st.sidebar.slider("Min AI Confidence", 0.1, 1.0, 0.25, 0.05)

if dry_run:
    st.warning("⚠️ DRY RUN ENABLED: No live orders will be placed.", icon="🚨")

# ✅ Monitor Controls
st.subheader("🟢 Monitor Service Controls")
pid_file = "/root/falah-ai-bot/monitor.pid"

def monitor_status():
    # Check if monitor process is running via PID file
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            # Check if process with this PID exists and is running
            if psutil.pid_exists(pid):
                p = psutil.Process(pid)
                if "python" in p.name() and "monitor_runner.py" in ' '.join(p.cmdline()):
                    return True
        except Exception:
            pass
    return False

status = "🟢 RUNNING" if monitor_status() else "🔴 STOPPED"
st.info(f"Monitor Status: **{status}**")

c1, c2, c3 = st.columns(3)

if c1.button("▶️ Start Monitor") and not monitor_status():
    proc = subprocess.Popen(
        ["nohup", "python3", "monitor_runner.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))
    st.success("✅ Monitor started.")
    st.rerun()

if c2.button("🟥 Stop Monitor") and monitor_status():
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        os.remove(pid_file)
        st.success("✅ Stopped Monitor.")
    except Exception as e:
        st.error(f"❌ Failed to stop monitor: {e}")
    st.rerun()

if c3.button("🔄 Run Once"):
    subprocess.run(["python3", "monitor_runner.py", "--once"])
    st.success("✅ Cycle complete.")
    
# ✅ Live Scanner
st.subheader("🔍 Auto Scanner")
if st.button("Scan Stocks"):
    st.info("⏳ Scanning...")
    scanned = run_smart_scan()
    st.session_state["scanned"] = scanned
    if not scanned.empty:
        st.success(f"✅ Found {len(scanned)} candidates.")
        st.dataframe(scanned.head(max_trades))

if "scanned" in st.session_state:
    scanned = st.session_state["scanned"].sort_values("Score", ascending=False).head(max_trades)

    if st.button("🚀 Place Orders"):
        kite = get_kite()
        if not validate_kite(kite):
            st.error("⚠️ Invalid access token."); st.stop()

        for _, row in scanned.iterrows():
            sym, cmp, rsi, atr, adx, ema10, ema21, volchg = row[["Symbol","CMP","RSI","ATR","ADX","EMA10","EMA21","VolumeChange"]]
            confidence = get_trade_probability(rsi, atr, adx, ema10, ema21, volchg)
            if confidence < min_conf: st.warning(f"❌ Skipped {sym} (Conf: {confidence:.2f})"); continue

            sl = compute_trailing_sl(cmp, atr)
            qty = calculate_quantity(capital, risk_pct, cmp, sl)
            if confidence >= 0.8: qty = int(qty * 1.3)

            msg = f"🚀 <b>{sym}</b>\nQty: {qty}\nEntry: ₹{cmp}\nSL: ₹{sl}\nConf: {confidence:.2f}"

        if dry_run:
            st.success(f"(Dry Run) {msg}")
            send_telegram(BOT_TOKEN, CHAT_ID, f"[DRY RUN]\n{msg}")
        elif is_market_open():
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
                log_trade_to_sheet(sym, qty, cmp, rsi, atr, adx, ai_score, action="BUY", exit_reason="", pnl="", outcome="")
                st.success(f"✅ Order placed for {sym}")
                send_telegram(BOT_TOKEN, CHAT_ID, msg)
        except Exception as e:
            st.error(f"❌ {sym} failed: {e}")
        
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

# ✅ Bot Controls
st.subheader("⚙️ Bot Controls")
b1, b2, b3 = st.columns(3)
if b1.button("📥 Fetch Historical"): fetch_all_historical(); st.success("✅ Historical fetched.")
if b2.button("▶️ Start Websockets"): start_all_websockets(); st.success("✅ Live Feed Started.")
if b3.button("🛑 Stop Websockets"): st.info("❌ Stop WS not implemented.")

if os.path.exists("/root/falah-ai-bot/last_fetch.txt"):
    st.info(f"Last Fetch: {open('/root/falah-ai-bot/last_fetch.txt').read().strip()}")
