from kiteconnect import KiteTicker
import threading
import json

live_prices = {}

def start_websocket(api_key, access_token, tokens):
    kws = KiteTicker(api_key, access_token)

    def on_ticks(ws, ticks):
        for tick in ticks:
            token = tick["instrument_token"]
            ltp = tick.get("last_price")
            if ltp:
                live_prices[token] = ltp

    def on_connect(ws, response):
        print("✅ WebSocket connected.")
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_LTP, tokens)

    def on_close(ws, code, reason):
        print("🔌 WebSocket closed:", reason)

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close

    # Start WebSocket in its own thread
    threading.Thread(target=kws.connect, daemon=True).start()
