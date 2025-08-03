# ws_worker.py

import time, json
from kiteconnect import KiteTicker
from credentials import load_secrets

# Load secrets
secrets = load_secrets()
api_key = secrets["zerodha"]["api_key"]
access_token = secrets["zerodha"]["access_token"]

# Load tokens from file or list
TOKENS_PATH = "/root/falah-ai-bot/ws_tokens.json"
try:
    with open(TOKENS_PATH, "r") as f:
        tokens = json.load(f)["tokens"]
except Exception as e:
    print("❌ Failed to load tokens:", e)
    exit(1)

# Websocket connect
kws = KiteTicker(api_key, access_token)

def on_ticks(ws, ticks):
    print("📈 Ticks:", ticks)

def on_connect(ws, response):
    print("✅ Connected.")
    ws.subscribe(tokens)

def on_close(ws, code, reason):
    print("🔌 Connection closed:", reason)

kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

print(f"🔌 Connecting to tokens: {tokens}")
kws.connect(threaded=True)

while True:
    time.sleep(1)
