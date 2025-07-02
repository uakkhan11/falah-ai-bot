# run_monitor.py

import time
import json
import toml
from kiteconnect import KiteConnect
from ws_live_prices import start_websockets, live_prices
from monitor_core import monitor_once

def log(msg):
    print(msg, flush=True)

# Load credentials
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)

API_KEY = secrets["zerodha"]["api_key"]

# Load access token
with open("/root/falah-ai-bot/access_token.json") as f:
    access_token = json.load(f)["access_token"]

print("✅ Using access_token:", access_token[:6], "...")

# Init Kite
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(access_token)

# Load tokens
with open("/root/falah-ai-bot/tokens.json") as f:
    token_map = json.load(f)
token_list = [int(t) for t in token_map.values()]

# Start WebSocket batches
start_websockets(API_KEY, access_token, token_list)

log("✅ WebSocket started.")

# Monitor loop
while True:
    monitor_once(kite, token_map, log, live_prices)
    time.sleep(900)
