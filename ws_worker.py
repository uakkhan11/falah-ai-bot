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
if len(sys.argv) != 4:
    print("Usage: python ws_worker.py <api_key> <access_token> <comma_separated_tokens>")
    sys.exit(1)

api_key = sys.argv[1]
access_token = sys.argv[2]
tokens = [int(t) for t in sys.argv[3].split(",")]

# --------- FILE PATHS ----------
output_file = "/tmp/live_prices_batch.json"

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
        print(f"[{datetime.now()}] ✅ Live prices written: {len(live_data)} symbols.")
        print(f"[{datetime.now()}] ✅ Prices saved to {output_file}")
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Error writing prices: {e}")


# --------- CLEANUP OLD FILES ----------
def clean_old_files():
    now = time.time()
    for f in os.listdir("/tmp"):
        if f.startswith("live_prices_") and f.endswith(".json"):
            path = os.path.join("/tmp", f)
            if now - os.path.getmtime(path) > 600:
                try:
                    os.remove(path)
                    print(f"[{datetime.now()}] 🗑️ Removed stale file: {f}")
                except Exception as e:
                    print(f"[{datetime.now()}] ⚠️ Error deleting {f}: {e}")


def schedule_cleanup():
    clean_old_files()
    Timer(600, schedule_cleanup).start()


# --------- KITE TICKER HANDLERS ----------
def on_ticks(ws, ticks):
    global last_write_time
    for t in ticks:
        live_data[t["instrument_token"]] = t["last_price"]
    if time.time() - last_write_time > write_interval_sec:
        write_to_file()


def on_connect(ws, response):
    if len(tokens) > 4000:
        print(f"[{datetime.now()}] ⚠️ Token limit exceeded: {len(tokens)} > 4000. Trimming list.")
        ws.subscribe(tokens[:4000])
        ws.set_mode(ws.MODE_FULL, tokens[:4000])
    else:
        print(f"[{datetime.now()}] ✅ Connected to WebSocket. Subscribing {len(tokens)} tokens.")
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)


def on_close(ws, code, reason):
    print(f"[{datetime.now()}] 🔴 WebSocket closed ({reason}). Reconnecting in 5s...")
    time.sleep(5)
    reconnect()


def on_error(ws, code, reason):
    print(f"[{datetime.now()}] ⚠️ WebSocket error: {reason}")


def reconnect():
    global kws
    try:
        kws.close()
    except:
        pass
    time.sleep(2)
    kws = KiteTicker(api_key, access_token)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    kws.connect(threaded=False)


# --------- MAIN EXECUTION ----------
print(f"[{datetime.now()}] 🔄 Starting WebSocket worker...")

if is_market_open():
    schedule_cleanup()
    kws = KiteTicker(api_key, access_token)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    kws.connect(threaded=False)
else:
    print(f"[{datetime.now()}] 🔒 Market is closed. Exiting WebSocket worker.")
