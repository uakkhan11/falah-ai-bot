# dashboard.py - Mobile & Trade-Friendly Falāh Bot Dashboard
import os, json, subprocess, signal, base64, psutil
from datetime import datetime
from pytz import timezone

import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
import joblib

# === Local imports ===
from credentials import load_secrets, get_kite, validate_kite
from data_fetch import fetch_all_historical
from fetch_intraday_data import fetch_intraday_data
from fetch_historical_batch import fetch_all_historical
from intraday_scanner import run_intraday_scan
from smart_scanner import run_smart_scan
from stock_analysis import analyze_stock, get_regime
from bulk_analysis import analyze_multiple_stocks
from ws_live_prices import start_all_websockets
from telegram_utils import send_telegram
from sheets import log_trade_to_sheet
from live_price_reader import get_symbol_price_map

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Falāh Bot Dashboard",
    layout="wide",
    page_icon="🌙"
)

# === Background image ===
def set_bg(image_path="icon-512.png"):
    if not os.path.exists(image_path): return
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
    /* Mobile Friendly Tweaks */
    @media (max-width: 768px) {{
        .block-container {{ padding: 0.5rem; }}
        button[kind="secondary"] {{ font-size: 18px; padding: 10px; }}
        .stDataFrame {{ font-size: 14px; }}
    }}
    </style>
    """, unsafe_allow_html=True)
set_bg()

# === Title Header ===
st.markdown("""
<div style='text-align: center; margin-top: -30px;'>
<h1>🟢 <b>Falāh Trading Bot</b></h1>
<h4 style='color: #6b7280;'>Ethical • Intelligent • Profitable</h4>
</div>
""", unsafe_allow_html=True)

# === Cached model load ===
@st.cache_resource
def load_model():
    return joblib.load("/root/falah-ai-bot/model.pkl")
model = load_model()

# === Utility functions ===
def compute_trailing_sl(cmp, atr, mult=1.5): return round(cmp - atr * mult, 2)
def calculate_quantity(capital, risk_pct, entry, sl):
    per_share_risk = entry - sl
    if per_share_risk <= 0: raise ValueError("SL must be below entry.")
    return max(int((capital * risk_pct) / per_share_risk), 1)
def is_market_open():
    now = datetime.now(timezone("Asia/Kolkata"))
    return now.weekday() < 5 and (9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30))

# === Load valid symbols ===
with open("symbol_to_token.json") as f:
    all_symbols = list(json.load(f).keys())
valid_symbols = sorted([s for s in all_symbols if s.isupper() and s.isalpha() and len(s) >= 3])

# === Load secrets ===
secrets = load_secrets()
BOT_TOKEN = secrets["telegram"]["bot_token"]
CHAT_ID = secrets["telegram"]["chat_id"]

# ========== SIDEBAR SETTINGS ==========
st.sidebar.header("⚙️ Capital & Trade Settings")
capital = st.sidebar.number_input("Daily Capital (₹)", 1000, 10_00_000, 100_000, 5000)
max_trades = st.sidebar.slider("Max Trades", 1, 10, 5)
dry_run = st.sidebar.toggle("Dry Run Mode", value=True)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
min_conf = st.sidebar.slider("Min AI Confidence", 0.1, 1.0, 0.25, 0.05)
if dry_run:
    st.sidebar.warning("🚨 DRY RUN ENABLED: No live orders will be placed.")

# ========== ACCESS TOKEN MANAGEMENT ==========
with st.expander("🔑 Access Token Management"):
    st.subheader("Generate New Access Token")
    kite = KiteConnect(api_key=secrets["zerodha"]["api_key"])
    st.markdown(f"[🔗 Login to Zerodha]({kite.login_url()})")
    request_token = st.text_input("Paste request_token here")
    if st.button("Generate Access Token"):
        try:
            data = kite.generate_session(request_token, api_secret=secrets["zerodha"]["api_secret"])
            access_token = data["access_token"]
            secrets_path = "/root/falah-ai-bot/secrets.json"
            with open(secrets_path, "r+") as f:
                sdata = json.load(f)
                sdata["zerodha"]["access_token"] = access_token
                f.seek(0); json.dump(sdata, f, indent=2); f.truncate()
            with open("/root/falah-ai-bot/access_token.json", "w") as f:
                json.dump({"access_token": access_token}, f)
            st.success("✅ Access token saved!")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ========== MONITOR CONTROLS ==========
pid_file = "/root/falah-ai-bot/monitor.pid"
def monitor_running():
    if not os.path.exists(pid_file): return False
    try:
        with open(pid_file) as f: pid = int(f.read().strip())
        if psutil.pid_exists(pid):
            p = psutil.Process(pid)
            return "monitor_runner.py" in " ".join(p.cmdline())
    except: pass
    return False

st.subheader("🟢 Monitor Controls")
cols = st.columns(3)
if cols[0].button("▶️ Start Monitor") and not monitor_running():
    proc = subprocess.Popen(["nohup", "python3", "monitor_runner.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(pid_file, "w") as f: f.write(str(proc.pid))
    st.success("✅ Monitor started."); st.rerun()
if cols[1].button("🟥 Stop Monitor") and monitor_running():
    try:
        with open(pid_file) as f: os.kill(int(f.read().strip()), signal.SIGTERM)
        os.remove(pid_file); st.success("✅ Monitor stopped.")
    except Exception as e: st.error(f"❌ {e}")
if cols[2].button("🔄 Run Once"):
    subprocess.run(["python3", "monitor_runner.py", "--once"])
    st.success("✅ Run complete.")

# ========== LIVE SCANNER ==========
st.subheader("🔍 Live Scanners")
if st.button("🔁 Run Intraday Scan"):
    with st.spinner("Scanning..."):
        results, logs = run_intraday_scan()
    if not results.empty:
        st.success(f"✅ {len(results)} stock(s) passed.")
        st.dataframe(results)
        st.download_button("⬇ Download", results.to_csv(index=False), file_name="intraday_results.csv")
    else:
        st.warning("⚠️ No stocks passed.")
    with st.expander("📋 Debug Logs"):
        st.text("\n".join(logs))

if st.button("🔎 Run Smart Scan"):
    with st.spinner("Running smart scan..."):
        scanned_df, debug_logs = run_smart_scan()
    if scanned_df.empty:
        st.warning("⚠️ No candidates found.")
    else:
        st.success(f"✅ Found {len(scanned_df)} candidates.")
        st.session_state["scanned"] = scanned_df
        st.dataframe(scanned_df.head(max_trades))

# === Show scanned results with order button ===
if "scanned" in st.session_state and not st.session_state["scanned"].empty:
    scanned_sorted = st.session_state["scanned"].sort_values("Score", ascending=False).head(max_trades)
    st.dataframe(scanned_sorted)
    if st.button("🚀 Place Orders"):
        kite = get_kite()
        if not validate_kite(kite):
            st.error("⚠️ Invalid token."); st.stop()
        for _, row in scanned_sorted.iterrows():
            sym, cmp, conf = row["symbol"], row["ltp"], row["Score"]
            if conf < min_conf:
                st.warning(f"⏩ Skipped {sym} (Conf: {conf:.2f})"); continue
            sl = round(cmp * 0.985, 2)
            try: qty = calculate_quantity(capital, risk_pct, cmp, sl)
            except ValueError as e: st.warning(f"{sym}: {e}"); continue
            msg = f"🚀 {sym} | Qty: {qty} | Entry: ₹{cmp} | SL: ₹{sl} | Conf: {conf:.2f}"
            if dry_run:
                st.info(f"[DRY RUN] {msg}"); send_telegram(BOT_TOKEN, CHAT_ID, "[DRY RUN]\n"+msg)
            elif is_market_open():
                try:
                    oid = kite.place_order(
                        variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NSE,
                        tradingsymbol=sym, transaction_type=kite.TRANSACTION_TYPE_BUY,
                        quantity=qty, order_type=kite.ORDER_TYPE_MARKET, product=kite.PRODUCT_CNC
                    )
                    st.success(f"✅ Order for {sym} | ID: {oid}")
                    send_telegram(BOT_TOKEN, CHAT_ID, msg)
                    log_trade_to_sheet(sym, qty, cmp, row.get("rsi"), None, None, conf, "BUY", "", "", "")
                except Exception as e:
                    st.error(f"❌ {sym}: {e}")

# ========== MANUAL LOOKUP ==========
st.subheader("🔍 Manual Lookup")
sym_in = st.selectbox("Select NSE Symbol", valid_symbols)
if st.button("Fetch Data"):
    kite = get_kite()
    if not validate_kite(kite): st.error("Invalid token."); st.stop()
    try:
        data = analyze_stock(kite, sym_in)
        st.write(f"✅ CMP: ₹{data['cmp']}")
        st.write(f"ATR(14): {data['atr']}, ADX: {data['adx']} ({get_regime(data['adx'])})")
        st.write(f"RSI: {data['rsi']} ({data['rsi_percentile']*100:.1f}%)")
        st.write(f"Recommendation: **{data['recommendation']}**")
        st.dataframe(data["history"].tail(10))
    except Exception as e: st.error(f"❌ {e}")

# ========== BULK ANALYSIS ==========
st.subheader("📊 Bulk Analysis")
bulk_syms = st.text_area("Enter symbols (comma-separated)").strip().upper()
if st.button("Analyze Bulk"):
    syms = [s.strip() for s in bulk_syms.split(",") if s.strip()]
    invalid = [s for s in syms if s not in valid_symbols]
    if invalid: st.error(f"Invalid: {', '.join(invalid)}")
    else:
        kite = get_kite()
        if not validate_kite(kite): st.error("Invalid token."); st.stop()
        results = analyze_multiple_stocks(kite, syms)
        df = pd.DataFrame(results)
        st.dataframe(df)

# ========== BOT UTILITIES ==========
st.subheader("⚙️ Bot Utilities")
u1, u2, u3 = st.columns(3)
if u1.button("📥 Fetch Historical"): fetch_all_historical(); st.success("✅ Done.")
if u2.button("▶️ Start Websockets"): start_all_websockets(); st.success("✅ Started.")
if u3.button("🛑 Stop Websockets"): st.info("Stop WS not implemented.")

# ========== INTRADAY FETCH ==========
st.subheader("🕐 Fetch Intraday Data")
interval = st.selectbox("Timeframe", ["15minute", "60minute"])
days = st.slider("Past Days", 1, 10, 5)
if st.button("📥 Fetch Now"):
    try:
        fetch_intraday_data(valid_symbols, interval=interval, days=days)
        st.success("✅ Data fetched.")
    except Exception as e:
        st.error(f"❌ {e}")
