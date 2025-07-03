# ws_worker.py

import sys
import json
import time
import os
from kiteconnect import KiteTicker

api_key = sys.argv[1]
access_token = sys.argv[2]
tokens = [int(t) for t in sys.argv[3].split(",")]

kws = KiteTicker(api_key, access_token)
live_prices = {}
last_save_time = time.time()

def on_connect(ws, response):
    print(f"✅ Connected. Subscribing {len(tokens)} tokens.")
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)

def on_ticks(ws, ticks):
    global last_save_time
    for tick in ticks:
        token = tick["instrument_token"]
        ltp = tick["last_price"]
        live_prices[token] = ltp
    if time.time() - last_save_time > 3:
        out = f"/tmp/live_prices_{os.getpid()}.json"
        with open(out, "w") as f:
            json.dump(live_prices, f)
        last_save_time = time.time()

def on_close(ws, code, reason):
    print(f"❌ Closed: {reason}")

def on_error(ws, error):
    print(f"⚠️ Error: {error}")

kws.on_connect = on_connect
kws.on_ticks = on_ticks
kws.on_close = on_close
kws.on_error = on_error
kws.connect(threaded=False)
