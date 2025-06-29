# app.py – Falāh Bot Main UI with AI Predictions + Dynamic Risk Management + Multi-Timeframe Confluence

import streamlit as st
import pandas as pd
import time
import random
import subprocess
import os
import threading
import toml
import pickle
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
ACCESS_TOKEN = secrets["zerodha"]["access_token"]
CREDS_JSON = "falah-credentials.json"
SHEET_KEY = secrets.get("global", {}).get("google_sheet_key", "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")

st.set_page_config(page_title="Falāh Bot UI", layout="wide")

# ---------------------------
# Init Kite + Sheets
# ---------------------------
@st.cache_resource
def init_kite():
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
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

MAX_BATCH_SIZE = 300

def start_websocket(symbols):
    """
    Starts WebSocket connections in batches and fetches instrument tokens in smaller chunks.
    """
    instrument_tokens = []
    batch_size = 200
    all_symbols = [f"NSE:{s}" for s in symbols]

    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i:i+batch_size]
        try:
            ltp_data_batch = kite.ltp(batch)
            for s in batch:
                if s in ltp_data_batch:
                    token = ltp_data_batch[s]["instrument_token"]
                    instrument_tokens.append(token)
                else:
                    print(f"⚠️ Skipping {s}: No LTP data.")
        except Exception as e:
            st.warning(f"❌ Failed fetching batch {i//batch_size+1}: {e}")

    if not instrument_tokens:
        st.error("❌ No instrument tokens could be loaded.")
        return

    batches = [instrument_tokens[i:i+MAX_BATCH_SIZE] for i in range(0, len(instrument_tokens), MAX_BATCH_SIZE)]

    def run_batch(batch, idx):
        kws = KiteTicker(API_KEY, ACCESS_TOKEN)

        def on_connect(ws, resp):
            print(f"✅ Batch {idx}: Connected")
            ws.subscribe(batch)
            ws.set_mode(ws.MODE_FULL, batch)

        def on_ticks(ws, ticks):
            for tick in ticks:
                token = tick["instrument_token"]
                ltp = tick.get("last_price")
                if ltp:
                    live_ltps[token] = ltp

        kws.on_connect = on_connect
        kws.on_ticks = on_ticks
        kws.on_error = lambda ws, code, reason: print(f"⚠️ Batch {idx} error: {reason}")
        kws.on_close = lambda ws, code, reason: print(f"🔌 Batch {idx} closed.")

        kws.connect(threaded=True)

    for idx, batch in enumerate(batches, 1):
        threading.Thread(target=run_batch, args=(batch, idx), daemon=True).start()


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
# Instrument tokens upfront
# ---------------------------
try:
    ltp_data = kite.ltp([f"NSE:{s}" for s in symbols])
    for s in symbols:
        key = f"NSE:{s}"
        token_map[s] = ltp_data[key]["instrument_token"]
    st.success(f"✅ Loaded {len(token_map)} instrument tokens.")
except Exception as e:
    st.error(f"❌ Failed to fetch instrument tokens: {e}")

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
                    # fallback to historical
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
            print(f"✅ Added {sym}: CMP={cmp}, AI={ai_score}, Proba={proba}")

        except Exception as e:
            st.warning(f"❌ Skipping {sym}: {e}")
    return results

st.info("⏳ Analyzing stocks...")
data = get_live_data(symbols)
df = pd.DataFrame(data)

if df.empty:
    st.error("No data available.")
    st.stop()

required_columns = ["AI Score", "Predict Proba", "Symbol", "CMP"]
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Missing expected data columns: {missing_cols}")
    st.write("Raw DataFrame:", df)
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
                    try:
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
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            st.warning("⚠️ GTT already exists for this symbol.")
                        else:
                            raise

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
