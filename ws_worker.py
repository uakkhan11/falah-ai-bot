# ws_worker.py

import json
import time
from kiteconnect import KiteTicker

# Load credentials from secrets.json
with open("/root/falah-ai-bot/secrets.json") as f:
    secrets = json.load(f)

api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]

# Load tokens from tokens.json
with open("/root/falah-ai-bot/tokens.json") as f:
    token_map = json.load(f)

tokens = [int(t) for t in token_map.values()]

# Initialize KiteTicker
kws = KiteTicker(api_key, access_token)

def on_connect(ws, response):
    print(f"✅ Connected. Subscribing to {len(tokens)} tokens...")
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)

def on_ticks(ws, ticks):
    for t in ticks:
        print(f"{t['instrument_token']} | LTP: {t['last_price']}")

def on_close(ws, code, reason):
    print(f"🔴 WebSocket closed: code={code}, reason={reason}")

def on_error(ws, code, reason):
    print(f"⚠️ WebSocket error: {reason}")

kws.on_connect = on_connect
kws.on_ticks = on_ticks
kws.on_close = on_close
kws.on_error = on_error

# Connect and run forever
print("🔄 Connecting to WebSocket...")
kws.connect(threaded=False)
