# ws_live_prices.py

from kiteconnect import KiteTicker

live_prices = {}

def start_websocket(api_key, access_token, tokens):
    kws = KiteTicker(api_key, access_token)

    def on_connect(ws, resp):
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_ticks(ws, ticks):
        for tick in ticks:
            live_prices[tick["instrument_token"]] = tick["last_price"]

    kws.on_connect = on_connect
    kws.on_ticks = on_ticks
    kws.connect(threaded=True)
