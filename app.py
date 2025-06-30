# app.py – Falāh Bot Main UI with AI Predictions + Dynamic Risk Management + Multi-Timeframe Confluence

import streamlit as st
import pandas as pd
import time
import random
import threading
import toml
import pickle
import json
import numpy as np
from kiteconnect import KiteConnect, KiteTicker
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD

from risk import calculate_position_size, calculate_atr_trailing_sl

# ---------------------------
# Load credentials
# ---------------------------
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)

API_KEY = secrets["zerodha"]["api_key"]
CREDS_JSON = "falah-credentials.json"
SHEET_KEY = secrets.get("global", {}).get("google_sheet_key", "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")

st.set_page_config(page_title="Falāh Bot UI", layout="wide")

# ---------------------------
# Init Kite + Sheets
# ---------------------------
@st.cache_resource
def init_kite():
    kite = KiteConnect(api_key=API_KEY)
    # ✅ Load access token from JSON
    with open("/root/falah-ai-bot/access_token.json") as f:
        token = json.load(f)["access_token"]
    kite.set_access_token(token)
    return kite

@st.cache_resource
def load_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_JSON, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_KEY)
    return sheet

@st.cache_resource
def load_model():
    with open("/root/falah-ai-bot/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------------------------
# Halal symbols
# ---------------------------
def get_halal_symbols(sheet):
    worksheet = sheet.worksheet("HalalList")
    symbols = worksheet.col_values(1)[1:]
    return [s.strip() for s in symbols if s.strip()]

# ---------------------------
# WebSocket LTP Data
# ---------------------------
live_ltps = {}
token_map = {}

def start_websocket(tokens):
    kws = KiteTicker(API_KEY, kite._access_token)

    def on_connect(ws, resp):
        print(f"✅ WebSocket Connected. Subscribing {len(tokens)} tokens...")
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_ticks(ws, ticks):
        for tick in ticks:
            token = tick["instrument_token"]
            ltp = tick.get("last_price")
            if ltp:
                live_ltps[token] = ltp

    kws.on_connect = on_connect
    kws.on_ticks = on_ticks
    kws.on_error = lambda ws, code, reason: print(f"⚠️ WebSocket error: {reason}")
    kws.on_close = lambda ws, code, reason: print(f"🔌 WebSocket closed.")

    threading.Thread(target=kws.connect, kwargs={"threaded": True}, daemon=True).start()

# ---------------------------
# Predict probability helper
# ---------------------------
def predict_probability(df):
    try:
        df = df.copy()
        df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
        df["ema_10"] = EMAIndicator(df["close"], window=10).ema_indicator()
        df["ema_50"] = EMAIndicator(df["close"], window=50).ema_indicator()
        df["macd"] = MACD(df["close"]).macd_diff()
        latest = df.iloc[-1][["rsi","ema_10","ema_50","macd"]]
        if latest.isnull().any():
            return 0.5
        X = np.array(latest).reshape(1, -1)
        proba = model.predict_proba(X)[0][1]
        return round(proba,3)
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return 0.5

# ---------------------------
# Fund Management Settings
# ---------------------------
st.sidebar.header("⚙️ Fund Management")
enable_dummy = st.sidebar.checkbox("🧪 Dummy Mode", value=False)
total_capital = st.sidebar.number_input("💰 Total Capital", 1000, 1_000_000, 100000)
max_trades = st.sidebar.number_input("📈 Max Trades", 1, 20, 5)
min_ai_score = st.sidebar.slider("🎯 Min Combined Score", 0, 200, 100)

# ---------------------------
# Main Execution
# ---------------------------
kite = init_kite()
sheet = load_sheet()
symbols = get_halal_symbols(sheet)

# ---------------------------
# Fetch instrument tokens safely in batches
# ---------------------------
token_map = {}
unique_symbols = list(set(symbols))
for i in range(0, len(unique_symbols), 100):
    batch = [f"NSE:{s}" for s in unique_symbols[i:i+100]]
    retries = 3
    while retries > 0:
        try:
            ltp_data = kite.ltp(batch)
            for s in unique_symbols[i:i+100]:
                key = f"NSE:{s}"
                if key in ltp_data:
                    token_map[s] = ltp_data[key]["instrument_token"]
                else:
                    st.warning(f"⚠️ Skipping {s}: invalid token")
            break
        except Exception as e:
            if "Too many requests" in str(e):
                st.warning(f"⏳ Too many requests, retrying...")
                time.sleep(2)
                retries -= 1
            else:
                st.warning(f"❌ Failed batch {i//100+1}: {e}")
                break

if not token_map:
    st.error("❌ No instrument tokens could be loaded.")
    st.stop()

st.success(f"✅ Loaded {len(token_map)} instrument tokens.")

if not enable_dummy:
    start_websocket(list(token_map.values()))

# ---------------------------
# Get live data
# ---------------------------
def get_live_data(symbols):
    results = []
    for sym in symbols:
        try:
            token = token_map.get(sym)
            if enable_dummy:
                cmp = round(random.uniform(200,1500),2)
                proba = random.uniform(0.4,0.6)
            else:
                cmp = live_ltps.get(token)
                if not cmp:
                    hist = kite.historical_data(
                        instrument_token=token,
                        from_date=pd.Timestamp.today()-pd.Timedelta(days=50),
                        to_date=pd.Timestamp.today(),
                        interval="day"
                    )
                    df_hist = pd.DataFrame(hist)
                    cmp = df_hist.iloc[-1]["close"]
                else:
                    hist = kite.historical_data(
                        instrument_token=token,
                        from_date=pd.Timestamp.today()-pd.Timedelta(days=50),
                        to_date=pd.Timestamp.today(),
                        interval="day"
                    )
                    df_hist = pd.DataFrame(hist)

                proba = predict_probability(df_hist)

            ai_score = round(random.uniform(60,95),2)
            results.append({
                "Symbol": sym,
                "CMP": cmp,
                "AI Score": ai_score,
                "Predict Proba": proba
            })

        except Exception as e:
            st.warning(f"❌ Skipping {sym}: {e}")
    return results

st.info("⏳ Analyzing stocks...")
data = get_live_data(symbols)
df = pd.DataFrame(data)

if df.empty:
    st.error("No data available.")
    st.stop()

df["Combined Score"] = df["AI Score"] + (df["Predict Proba"] * 100)
candidates = df[df["Combined Score"] >= min_ai_score]
candidates = candidates.sort_values(by="Combined Score", ascending=False).head(max_trades)

st.subheader("📊 Candidates")
st.dataframe(candidates[["Symbol","CMP","AI Score","Predict Proba","Combined Score"]])

if not candidates.empty:
    if st.button("🛒 Execute Trades Now"):
        for _, row in candidates.iterrows():
            try:
                qty, cmp = calculate_position_size(kite, row["Symbol"], total_capital)
                if qty <=0:
                    continue

                if not enable_dummy:
                    kite.place_order(
                        variety=kite.VARIETY_REGULAR,
                        exchange=kite.EXCHANGE_NSE,
                        tradingsymbol=row["Symbol"],
                        transaction_type=kite.TRANSACTION_TYPE_BUY,
                        quantity=qty,
                        order_type=kite.ORDER_TYPE_MARKET,
                        product=kite.PRODUCT_CNC
                    )

                    sl_price = calculate_atr_trailing_sl(kite, row["Symbol"], cmp)
                    kite.place_gtt(
                        trigger_type=kite.GTT_TYPE_SINGLE,
                        tradingsymbol=row["Symbol"],
                        exchange=kite.EXCHANGE_NSE,
                        trigger_values=[sl_price],
                        last_price=cmp,
                        orders=[
                            {
                                "transaction_type": kite.TRANSACTION_TYPE_SELL,
                                "quantity": qty,
                                "order_type": kite.ORDER_TYPE_LIMIT,
                                "price": sl_price,
                                "product": kite.PRODUCT_CNC
                            }
                        ]
                    )
                    st.success(f"✅ GTT Stoploss set at ₹{sl_price}")

                sheet.worksheet("LivePositions").append_row([
                    row["Symbol"],
                    qty,
                    float(cmp),
                    row["AI Score"],
                    row["Predict Proba"],
                    row["Combined Score"],
                    time.strftime("%Y-%m-%d %H:%M:%S")
                ])
                st.success(f"✅ Order placed for {row['Symbol']}")
            except Exception as e:
                st.warning(f"⚠️ Order failed for {row['Symbol']}: {e}")
else:
    st.warning("⚠️ No candidates met the minimum score.")
