import threading
from kiteconnect import KiteConnect, KiteTicker
import logging
from datetime import datetime, time as dt_time
import pytz

class LivePriceStreamer:
    def __init__(self, kite: KiteConnect, tokens: list, timezone="Asia/Kolkata"):
        self.kite = kite
        self.tokens = tokens
        self.kws = None
        self.running = False
        self.prices = {}
        self.thread = None
        self.timezone = pytz.timezone(timezone)
        self.market_open = dt_time(9, 15)
        self.market_close = dt_time(15, 30)

    def _on_ticks(self, ws, ticks):
        for tick in ticks:
            sym = tick.get('tradingsymbol')
            self.prices[sym] = tick.get('last_price')

    def _on_connect(self, ws, response):
        ws.subscribe(self.tokens)
        ws.set_mode(ws.MODE_FULL, self.tokens)

    def _is_market_open(self):
        now = datetime.now(self.timezone).time()
        return self.market_open <= now <= self.market_close

    def _run(self):
        if not self._is_market_open():
            logging.info("Market is closed - skipping live price streaming.")
            return

        self.kws = KiteTicker(self.kite.api_key, self.kite.access_token)
        self.kws.on_ticks = self._on_ticks
        self.kws.on_connect = self._on_connect
        self.running = True
        while self.running and self._is_market_open():
            self.kws.connect()
        logging.info("Stopped live price streaming as market closed.")
        self.running = False

    def start(self):
        if self.running:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logging.info("Live price streamer started.")

    def stop(self):
        self.running = False
        if self.kws:
            self.kws.disconnect()
        if self.thread:
            self.thread.join()
        logging.info("Live price streamer stopped.")

    def get_price(self, symbol):
        return self.prices.get(symbol)
