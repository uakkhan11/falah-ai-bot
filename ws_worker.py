# ws_worker.py
import sys
import json
from kiteconnect import KiteTicker

if len(sys.argv) < 4:
    print("Usage: python3 ws_worker.py <api_key> <access_token> <token1,token2,...>")
    sys.exit(1)

api_key = sys.argv[1]
access_token = sys.argv[2]
tokens = [int(t) for t in sys.argv[3].split(",")]

print(f"✅ Worker started for {len(tokens)} tokens.")

live_prices = {}

kws = KiteTicker(api_key, access_token)

def on_connect(ws, resp):
    print(f"✅ Subscribing {len(tokens)} tokens...")
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)

def on_ticks(ws, ticks):
    for tick in ticks:
        live_prices[tick["instrument_token"]] = tick["last_price"]

def on_close(ws, code, reason):
    print(f"🔌 Closed: {code} - {reason}")

def on_error(ws, code, reason):
    print(f"⚠️ Error: {reason} (Code {code})")

kws.on_connect = on_connect
kws.on_ticks = on_ticks
kws.on_close = on_close
kws.on_error = on_error

kws.connect(threaded=False)
