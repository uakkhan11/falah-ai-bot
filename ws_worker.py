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

# --------- FILE PATHS ----------
output_file = f"/tmp/live_prices_{worker_index}.json"

# --------- LIVE PRICE DATA ----------
live_data = {}
last_write_time = time.time()
write_interval_sec = 5

# --------- TIMEZONE ----------
IST = pytz.timezone("Asia/Kolkata")

def is_market_open():
    now = datetime.now(IST).time()
    return dt_time(9, 15) <= now <= dt_time(15, 30)

# --------- FILE WRITE FUNCTION ----------
def write_to_file():
    global last_write_time
    try:
        with open(output_file, "w") as f:
            json.dump(live_data, f)
        last_write_time = time.time()
        print(f"[{datetime.now()}] ✅ Worker {worker_index} wrote {len(live_data)} prices to {output_file}")
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Worker {worker_index} error writing prices: {e}")

# --------- KITE TICKER HANDLERS ----------
def on_ticks(ws, ticks):
    global last_write_time
    for t in ticks:
        live_data[t["instrument_token"]] = t["last_price"]
    if time.time() - last_write_time > write_interval_sec:
        write_to_file()

def on_connect(ws, response):
    if len(tokens) > 4000:
        print(f"[{datetime.now()}] ⚠️ Worker {worker_index}: Token limit exceeded. Subscribing only first 4000.")
        ws.subscribe(tokens[:4000])
        ws.set_mode(ws.MODE_FULL, tokens[:4000])
    else:
        print(f"[{datetime.now()}] ✅ Worker {worker_index} connected. Subscribing {len(tokens)} tokens.")
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

def on_close(ws, code, reason):
    print(f"[{datetime.now()}] 🔴 Worker {worker_index} WebSocket closed ({reason}). Exiting.")
    os._exit(1)  # Clean exit; supervisor should restart

def on_error(ws, code, reason):
    print(f"[{datetime.now()}] ⚠️ Worker {worker_index} WebSocket error: {reason}")

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
