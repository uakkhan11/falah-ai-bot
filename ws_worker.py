# ws_worker.py

import sys
import time
import signal
from kiteconnect import KiteTicker
from credentials import load_secrets

# Load credentials from secrets.json
secrets = load_secrets()
api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]

# Validate arguments
if len(sys.argv) != 3:
    print("Usage: python ws_worker.py <comma_separated_tokens> <worker_index>")
    sys.exit(1)

# Parse arguments
tokens = [int(t) for t in sys.argv[1].split(",")]
worker_index = sys.argv[2]

# Setup KiteTicker
kws = KiteTicker(api_key, access_token)

def on_ticks(ws, ticks):
    print(f"[Worker {worker_index}] ✅ Ticks received: {ticks}")

def on_connect(ws, response):
    print(f"[Worker {worker_index}] ✅ Connected.")
    ws.subscribe(tokens)

def on_close(ws, code, reason):
    print(f"[Worker {worker_index}] ❌ Connection closed: {reason}")

def on_error(ws, code, reason):
    print(f"[Worker {worker_index}] ❌ Error: {reason}")

def on_reconnect(ws, attempts_count):
    print(f"[Worker {worker_index}] 🔁 Reconnecting... Attempt {attempts_count}")

def on_noreconnect(ws):
    print(f"[Worker {worker_index}] ❌ Reconnection failed.")

# Assign callbacks
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close
kws.on_error = on_error
kws.on_reconnect = on_reconnect
kws.on_noreconnect = on_noreconnect

# Signal handler for graceful shutdown
def exit_gracefully(signum, frame):
    print(f"[Worker {worker_index}] 🔌 Shutting down...")
    kws.close()
    sys.exit(0)

signal.signal(signal.SIGINT, exit_gracefully)
signal.signal(signal.SIGTERM, exit_gracefully)

print(f"[Worker {worker_index}] 🚀 Connecting to tokens: {tokens}")
kws.connect(threaded=True)

# Keep alive
while True:
    time.sleep(1)
