# run_monitor.py

import time
import json
from kiteconnect import KiteConnect
from ws_live_prices import start_websockets
from monitor_core import monitor_once

def log(msg):
    print(msg, flush=True)

# Load credentials
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "r") as f:
    import toml
    secrets = toml.load(f)

API_KEY = secrets["zerodha"]["api_key"]

# Load access token
with open("/root/falah-ai-bot/access_token.json") as f:
    access_token = json.load(f)["access_token"]

# Init Kite
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(access_token)

# Load tokens
with open("/root/falah-ai-bot/tokens.json") as f:
    token_map = json.load(f)
token_list = [int(t) for t in token_map.values()]

# Start WebSocket batches (no live_prices dict here)
start_websockets(API_KEY, access_token, token_list)

log("✅ WebSocket started.")

# Monitor loop
while True:
    # Note: We REMOVE live_prices argument
    monitor_once(kite, token_map, log)
    time.sleep(900)
