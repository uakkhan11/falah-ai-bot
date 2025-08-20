from kiteconnect import KiteTicker
from datetime import datetime
import pandas as pd
import threading
from config import Config  # Import your Config class

INTERVAL_TO_MINUTES = {
    "15minute": 15,
    "1hour": 60,
    "day": 1440
}

class LiveCandleAggregator:
    def __init__(self, tokens, interval="15minute"):
        try:
            self.interval = interval
            self.tokens = tokens  # List of instrument tokens as int
        
            # Load Config and authenticate automatically
            self.config = Config()
            self.config.authenticate()  # This loads or obtains access token
            print("[INFO] LiveCandleAggregator initialized.")
        except Exception as e:
            print(f"[ERROR] Exception during initialization: {e}")
            raise
        
        if not self.config.ACCESS_TOKEN:
            raise Exception("Failed to obtain access token. Cannot start websocket.")
        
        self.api_key = self.config.API_KEY
        self.access_token = self.config.ACCESS_TOKEN
        
        # Initialize KiteTicker with loaded token
        self.kws = KiteTicker(self.api_key, self.access_token)
        
        self._candles = {}  # (token, candle_id): candle dict
        self._mutex = threading.Lock()
        self._stopped = False
        
        # Attach event handlers
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_close = self.on_close
    
    def get_candle_id(self, ts):
        if self.interval == "day":
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        delta = INTERVAL_TO_MINUTES[self.interval]
        rounded_minute = (ts.minute // delta) * delta
        return ts.replace(minute=rounded_minute, second=0, microsecond=0)
    
    def on_ticks(self, ws, ticks):
        print(f"[DEBUG] Received {len(ticks)} tick(s) at {datetime.now()}")
        now = datetime.now()
        with self._mutex:
            for tick in ticks:
                token = tick['instrument_token']
                price = tick.get('last_price') or tick.get('average_price')
                if price is None:
                    print(f"[WARNING] Tick for token {token} has no price - skipping")
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
                    c["volume"] = volume 
        
    def on_connect(self, ws, response):
        print(f"[DEBUG] Connected to WebSocket, subscribing to tokens: {self.tokens}")
        ws.subscribe(self.tokens)
        ws.set_mode(ws.MODE_FULL, self.tokens)
        print(f"Subscribed to tokens: {self.tokens}")
    
    def on_close(self, ws, code, reason):
        self._stopped = True
        print(f"Websocket closed - Code: {code}, Reason: {reason}")
    
   def start(self):
        import time
        attempt = 0
        max_delay = 300  # 5 minutes max
    
        while True:
            try:
                print("[INFO] Attempting to start WebSocket connection...")
                self.kws.connect(threaded=True)
                break  # Success
            except Exception as e:
                wait = min(2 ** attempt, max_delay)
                print(f"[ERROR] WebSocket connection failed: {e}. Retrying in {wait} seconds...")
                time.sleep(wait)
                attempt += 1
    
    def stop(self):
        try:
            self.kws.close()
            self._stopped = True
            print("[INFO] Websocket closed successfully.")
        except Exception as e:
            print(f"[ERROR] Error closing KiteTicker websocket: {e}")
    
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
        import time
        while True:
            with self._mutex:
                if all(any(key[0] == t for key in self._candles) for t in self.tokens):
                    break
            time.sleep(0.2)
