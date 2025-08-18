from kiteconnect import KiteTicker
from datetime import datetime, timedelta
import pandas as pd
import threading

INTERVAL_TO_MINUTES = {
    "15minute": 15,
    "1hour": 60,
    "day": 1440
}

class LiveCandleAggregator:
    def __init__(self, api_key, access_token, tokens, interval="15minute"):
        self.interval = interval
        self.tokens = tokens  # List of instrument tokens as int
        self.api_key = api_key
        self.access_token = access_token

        self.kws = KiteTicker(api_key, access_token)
        self._candles = {}  # (token, candle_id): candle dict
        self._mutex = threading.Lock()
        self._stopped = False

        # Attach event handlers
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_close = self.on_close

    def get_candle_id(self, ts):
        # Round down to nearest interval (e.g. 09:32 → 09:30 for 15min)
        if self.interval == "day":
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        delta = INTERVAL_TO_MINUTES[self.interval]
        rounded_minute = (ts.minute // delta) * delta
        return ts.replace(minute=rounded_minute, second=0, microsecond=0)
    
    def on_ticks(self, ws, ticks):
        now = datetime.now()
        with self._mutex:
            for tick in ticks:
                token = tick['instrument_token']
                price = tick.get('last_price') or tick.get('average_price')
                if price is None:
                    continue
                volume = tick.get('volume', 0)
                ts = pd.Timestamp.now()
                candle_id = self.get_candle_id(ts)
                key = (token, candle_id)
                if key not in self._candles:
                    self._candles[key] = {
                        "open": price, "high": price, "low": price,
                        "close": price, "volume": volume,
                        "ts": candle_id,
                    }
                else:
                    c = self._candles[key]
                    c["high"] = max(c["high"], price)
                    c["low"] = min(c["low"], price)
                    c["close"] = price
                    c["volume"] = volume  # Replace with volume if monotonic, or c["volume"] += if incremental
        
    def on_connect(self, ws, response):
        ws.subscribe(self.tokens)
        ws.set_mode(ws.MODE_FULL, self.tokens)
        print(f"Subscribed to tokens: {self.tokens}")

    def on_close(self, ws, code, reason):
        self._stopped = True
        print("Websocket closed", code, reason)

    def start(self):
        self.kws.connect(threaded=True)

    def stop(self):
        self.kws.close()
        self._stopped = True

    def get_live_candle(self, token):
        with self._mutex:
            now = pd.Timestamp.now()
            candle_id = self.get_candle_id(now)
            return self._candles.get((token, candle_id))

    def get_all_live_candles(self):
        with self._mutex:
            latest = {}
            now = pd.Timestamp.now()
            candle_id = self.get_candle_id(now)
            for token in self.tokens:
                candle = self._candles.get((token, candle_id))
                if candle:
                    latest[token] = candle.copy()
            return latest

    def wait_until_ready(self):
        # Utility to wait for at least one tick received for each token
        import time
        while True:
            with self._mutex:
                if all(any(key[0]==t for key in self._candles) for t in self.tokens):
                    break
            time.sleep(0.2)
