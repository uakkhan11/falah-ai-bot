# ws_worker.py

import json
import time
import os
from kiteconnect import KiteTicker

# CONFIGURATION
SECRETS_PATH = "/root/falah-ai-bot/secrets.json"
TOKENS_PATH = "/root/falah-ai-bot/tokens.json"
OUTPUT_PATH = "/tmp/live_prices_{batch_id}.json"
BATCH_ID = os.getenv("BATCH_ID", "default")  # Allow optional batch separation
WRITE_INTERVAL = 3  # seconds

# LOAD SECRETS
with open(SECRETS_PATH) as f:
    secrets = json.load(f)

api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]

# LOAD TOKENS
with open(TOKENS_PATH) as f:
    token_map = json.load(f)

tokens = [int(t) for t in token_map.values()]
print(f"✅ Loaded {len(tokens)} tokens.")

# SHARED PRICE STORE
live_prices = {}

# FILE WRITE HANDLER
def save_live_prices():
    temp_path = f"{OUTPUT_PATH.format(batch_id=BATCH_ID)}.tmp"
    final_path = OUTPUT_PATH.format(batch_id=BATCH_ID)
    with open(temp_path, "w") as f:
        json.dump(live_prices, f)
    os.replace(temp_path, final_path)  # atomic write
    print(f"💾 Updated {final_path} with {len(live_prices)} prices.")

# KITE TICKER
kws = KiteTicker(api_key, access_token)

def on_connect(ws, response):
    print("✅ Connected to WebSocket, subscribing...")
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_QUOTE, tokens)

def on_ticks(ws, ticks):
    for tick in ticks:
        token = tick["instrument_token"]
        ltp = tick["last_price"]
        live_prices[str(token)] = ltp
    # Throttled file write
    global last_write
    now = time.time()
    if now - last_write > WRITE_INTERVAL:
        save_live_prices()
        last_write = now

def on_close(ws, code, reason):
    print(f"🔴 WebSocket closed. Code={code}, Reason={reason}")
    save_live_prices()

def on_error(ws, code, reason):
    print(f"⚠️ WebSocket error: Code={code}, Reason={reason}")

def on_order_update(ws, data):
    pass  # Optional, unused

# ATTACH HANDLERS
kws.on_connect = on_connect
kws.on_ticks = on_ticks
kws.on_close = on_close
kws.on_error = on_error
kws.on_order_update = on_order_update

# MAIN LOOP
if __name__ == "__main__":
    last_write = time.time()
    try:
        print("🔄 Starting WebSocket connection...")
        kws.connect(threaded=False)
    except Exception as e:
        print(f"❌ Exception: {e}")
        save_live_prices()
