# ws_worker.py

import sys
from kiteconnect import KiteTicker

api_key = sys.argv[1]
access_token = sys.argv[2]
tokens = [int(t) for t in sys.argv[3].split(",")]

kws = KiteTicker(api_key, access_token)

def on_connect(ws, response):
    print(f"✅ Connected: Subscribing {len(tokens)} tokens")
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)

def on_ticks(ws, ticks):
    # print(f"✅ Ticks received: {len(ticks)}")

def on_close(ws, code, reason):
    print(f"WebSocket closed: code={code}, reason={reason}")

def on_error(ws, code, reason):
    print(f"⚠️ Error: {reason}")

kws.on_connect = on_connect
kws.on_ticks = on_ticks
kws.on_close = on_close
kws.on_error = on_error

kws.connect(threaded=False)
