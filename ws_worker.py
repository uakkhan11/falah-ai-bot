# ws_worker.py

import json
import os
import sys
import time
from datetime import datetime, time as dt_time
from threading import Timer
from kiteconnect import KiteTicker
import pytz

# --------- ARGUMENT VALIDATION ----------
if len(sys.argv) != 5:
    print("Usage: python ws_worker.py <api_key> <access_token> <comma_separated_tokens> <worker_index>")
    sys.exit(1)

api_key = sys.argv[1]
access_token = sys.argv[2]
tokens = [int(t) for t in sys.argv[3].split(",")]
worker_index = sys.argv[4]

# --------- PATHS ----------
TMP_FILE = f"/tmp/live_prices_{worker_index}.json"
MERGED_FILE = "live_prices.json"
TOKEN_MAP_FILE = "instrument_token_map.json"  # Required for symbol ↔ token mapping

# --------- LIVE DATA CACHE ----------
live_data = {}
last_write_time = time.time()
write_interval_sec = 5

# --------- TIMEZONE ----------
IST = pytz.timezone("Asia/Kolkata")

# --------- HELPER ----------
def is_market_open():
    now = datetime.now(IST).time()
    return dt_time(9, 15) <= now <= dt_time(15, 30)

# --------- SAVE TO TEMP ----------
def write_temp_file():
    try:
        with open(TMP_FILE, "w") as f:
            json.dump(live_data, f)
        print(f"[{datetime.now()}] ✅ Worker {worker_index}: wrote {len(live_data)} to {TMP_FILE}")
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Worker {worker_index} error writing: {e}")

# --------- MERGE ALL WORKER FILES ----------
def merge_worker_files():
    merged = {}
    try:
        token_map = {}
        if os.path.exists(TOKEN_MAP_FILE):
            with open(TOKEN_MAP_FILE) as f:
                token_map = json.load(f)

        for file in os.listdir("/tmp/"):
            if file.startswith("live_prices_") and file.endswith(".json"):
                with open(os.path.join("/tmp/", file)) as f:
                    temp_data = json.load(f)
                    for token_str, price in temp_data.items():
                        token = int(token_str)
                        symbol = token_map.get(str(token), f"TOKEN_{token}")
                        merged[symbol] = round(float(price), 2)

        if merged:
            with open(MERGED_FILE, "w") as f:
                json.dump(merged, f, indent=2)
            print(f"[{datetime.now()}] 💾 Updated {MERGED_FILE} with {len(merged)} symbols")

    except Exception as e:
        print(f"[{datetime.now()}] ❌ Merge error: {e}")

# --------- KITE TICKER EVENTS ----------
def on_ticks(ws, ticks):
    global last_write_time
    for t in ticks:
        live_data[str(t["instrument_token"])] = t["last_price"]

    if time.time() - last_write_time > write_interval_sec:
        write_temp_file()
        merge_worker_files()
        last_write_time = time.time()

def on_connect(ws, response):
    if len(tokens) > 4000:
        print(f"[{datetime.now()}] ⚠️ Too many tokens. Subscribing only first 4000.")
        sub = tokens[:4000]
    else:
        sub = tokens

    print(f"[{datetime.now()}] ✅ Connected. Subscribing to {len(sub)} tokens.")
    ws.subscribe(sub)
    ws.set_mode(ws.MODE_FULL, sub)

def on_close(ws, code, reason):
    print(f"[{datetime.now()}] 🔴 Closed: {reason}. Exiting.")
    os._exit(1)

def on_error(ws, code, reason):
    print(f"[{datetime.now()}] ⚠️ WebSocket error: {reason}")

# --------- MAIN ----------
print(f"[{datetime.now()}] 🔄 Worker {worker_index} starting WebSocket...")

if is_market_open():
    kws = KiteTicker(api_key, access_token)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    kws.connect(threaded=False)
else:
    print(f"[{datetime.now()}] 🔒 Market is closed. Worker {worker_index} exiting.")
