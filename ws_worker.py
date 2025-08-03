# ws_worker.py

import sys
import time
import json
import os
from kiteconnect import KiteTicker
from credentials import load_secrets

# Check args: only worker index is required
if len(sys.argv) != 2:
    print("Usage: python ws_worker.py <worker_index>")
    sys.exit(1)

worker_index = sys.argv[1]
token_file = f"/root/falah-ai-bot/ws_tokens/ws_tokens_{worker_index}.json"

if not os.path.exists(token_file):
    print(f"❌ Token file not found: {token_file}")
    sys.exit(1)

with open(token_file, "r") as f:
    data = json.load(f)
    tokens = data.get("tokens", [])
    if not tokens:
        print(f"❌ No tokens found in {token_file}")
        sys.exit(1)

# Load secrets
secrets = load_secrets()
api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]

# Initialize KiteTicker
kws = KiteTicker(api_key, access_token)

def on_ticks(ws, ticks):
    print(f"[Worker {worker_index}] ✅ Ticks received: {len(ticks)}")

def on_connect(ws, response):
    print(f"[Worker {worker_index}] 🔌 Connected. Subscribing to {len(tokens)} tokens.")
    ws.subscribe(tokens)

def on_close(ws, code, reason):
    print(f"[Worker {worker_index}] 🔌 Connection closed: {reason}")

kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

print(f"[Worker {worker_index}] 🚀 Connecting to Kite WebSocket with {len(tokens)} tokens")
kws.connect(threaded=True)

# Keep alive
while True:
    time.sleep(1)
