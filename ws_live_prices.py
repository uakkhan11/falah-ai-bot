# ws_live_prices.py

from kiteconnect import KiteTicker

# Global dictionary to store live prices
live_prices = {}

def start_websockets(api_key, access_token, tokens, batch_size=300):
    """
    Starts multiple KiteTicker WebSocket connections in batches.
    """
    import threading

    batches = [tokens[i:i + batch_size] for i in range(0, len(tokens), batch_size)]
    print(f"✅ Splitting tokens into {len(batches)} batch(es).")

    for i, batch in enumerate(batches, start=1):
        kws = KiteTicker(api_key, access_token)

        def on_connect(ws, resp, batch=batch, idx=i):
            print(f"✅ Batch {idx}: Connected. Subscribing {len(batch)} tokens...")
            ws.subscribe(batch)
            ws.set_mode(ws.MODE_FULL, batch)

        def on_ticks(ws, ticks):
            for tick in ticks:
                live_prices[tick["instrument_token"]] = tick["last_price"]

        def on_close(ws, code, reason):
            print(f"🔌 Batch {i} closed: {code} - {reason}")

        def on_error(ws, code, reason):
            print(f"⚠️ Batch {i} error: {reason} (Code {code})")

        kws.on_connect = on_connect
        kws.on_ticks = on_ticks
        kws.on_close = on_close
        kws.on_error = on_error

        # Launch each WebSocket in its own thread
        threading.Thread(
            target=kws.connect,
            kwargs={"threaded": True},
            daemon=True
        ).start()
