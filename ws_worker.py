# ws_worker.py

import sys
import time
from kiteconnect import KiteTicker
from credentials import load_secrets

# Load credentials
secrets = load_secrets()
api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]

# Validate args
if len(sys.argv) != 3:
    print("Usage: python ws_worker.py <comma_separated_tokens> <worker_index>")
    sys.exit(1)

tokens = [int(t) for t in sys.argv[1].split(",")]
worker_index = sys.argv[2]

# Connect
kws = KiteTicker(api_key, access_token)

def on_ticks(ws, ticks):
    print(f"[{worker_index}] Ticks: {ticks}")

def on_connect(ws, response):
    print(f"[{worker_index}] Connected.")
    ws.subscribe(tokens)

def on_close(ws, code, reason):
    print(f"[{worker_index}] Connection closed. {reason}")

kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

print(f"[{worker_index}] Connecting to tokens: {tokens}")
kws.connect(threaded=True)

while True:
    time.sleep(1)
