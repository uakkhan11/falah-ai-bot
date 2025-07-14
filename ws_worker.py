# ws_worker.py

import json
import os
import time
from datetime import datetime
from kiteconnect import KiteTicker
from threading import Timer

# Load credentials
with open("/root/falah-ai-bot/secrets.json") as f:
    secrets = json.load(f)
api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]

# Load tokens
with open("/root/falah-ai-bot/tokens.json") as f:
    token_map = json.load(f)
tokens = [int(t) for t in token_map.values()]

output_file = "/tmp/live_prices_batch.json"
live_data = {}
last_write_time = time.time()

kws = KiteTicker(api_key, access_token)

def write_to_file():
    global last_write_time
    with open(output_file, "w") as f:
        json.dump(live_data, f)
    last_write_time = time.time()
    print(f"[{datetime.now()}] ✅ Live prices written: {len(live_data)} symbols.")
    print(f"[{datetime.now()}] ✅ Prices saved to {output_file}")

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

def on_ticks(ws, ticks):
    for t in ticks:
        live_data[t["instrument_token"]] = t["last_price"]
    if time.time() - last_write_time > 2:
        write_to_file()

def on_connect(ws, response):
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
    kws.close()
    time.sleep(2)
    kws = KiteTicker(api_key, access_token)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error
    kws.connect(threaded=False)

kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close
kws.on_error = on_error

print(f"[{datetime.now()}] 🔄 Starting WebSocket worker...")
schedule_cleanup()
kws.connect(threaded=False)
