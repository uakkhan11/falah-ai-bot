# ws_live_prices.py

from kiteconnect import KiteTicker

# Dictionary to hold live LTPs
live_prices = {}

def start_websocket(api_key, access_token, tokens):
    kws = KiteTicker(api_key, access_token)

    def on_connect(ws, resp):
        print(f"✅ WebSocket connected. Subscribing {len(tokens)} tokens...")
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_ticks(ws, ticks):
        for tick in ticks:
            live_prices[tick["instrument_token"]] = tick["last_price"]

    def on_close(ws, code, reason):
        print(f"🔌 WebSocket closed. Code: {code}, Reason: {reason}")

    def on_error(ws, code, reason):
        print(f"⚠️ WebSocket error: {reason} (Code {code})")

    kws.on_connect = on_connect
    kws.on_ticks = on_ticks
    kws.on_close = on_close
    kws.on_error = on_error

    # Start in a background thread
    kws.connect(threaded=True)
