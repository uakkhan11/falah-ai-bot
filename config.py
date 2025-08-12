# config.py

import os
import json
import logging
from auto_auth import auto_authenticate  # Local Redirect Server auth helper

TOKENS_FILE = "kite_tokens.json"

class Config:
    def __init__(self):
        # --- Trading Parameters ---
        self.MAX_POSITION_LOSS_PCT = 0.03
        self.INITIAL_CAPITAL = 1_000_000
        self.POSITION_SIZE = 100_000
        self.RISK_PER_TRADE = 0.01
        self.PROFIT_TARGET = 0.10
        self.ATR_SL_MULT = 2.8
        self.ATR_PERIOD = 14
        self.TRAIL_TRIGGER = 0.07
        self.TRAIL_DISTANCE = 0.03
        self.ADX_THRESHOLD_DEFAULT = 15
        self.MAX_POSITIONS = 5
        self.DAILY_LOSS_LIMIT_PCT = 0.05

        # --- Zerodha API credentials ---
        self.API_KEY = "your_api_key_here"
        self.API_SECRET = "your_api_secret_here"

        # --- Telegram credentials ---
        self.TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
        self.TELEGRAM_CHAT_ID = "your_telegram_chat_id"

        # Access token placeholder
        self.ACCESS_TOKEN = None

        # Holds KiteConnect client after authentication
        self.kite = None

        # Logging configuration
        logging.basicConfig(level=logging.INFO)

    def authenticate(self):
        """
        Authenticate using local redirect server flow from auto_auth.py
        and set the authenticated KiteConnect instance.
        """
        kite = auto_authenticate()
        if not kite:
            raise Exception("Authentication failed.")
        self.kite = kite

    # Optional: legacy helpers in case you want to manage token file manually.
    def _load_saved_token(self):
        """Load saved access token from file if needed."""
        if os.path.exists(TOKENS_FILE):
            try:
                with open(TOKENS_FILE, "r") as f:
                    data = json.load(f)
                    token = data.get("access_token")
                    if token:
                        self.ACCESS_TOKEN = token
                        logging.info("Loaded ACCESS_TOKEN from file.")
            except Exception as e:
                logging.warning(f"Could not load saved token: {e}")

    def _save_token(self):
        """Save access token to file."""
        try:
            with open(TOKENS_FILE, "w") as f:
                json.dump({"access_token": self.ACCESS_TOKEN}, f)
            logging.info("ACCESS_TOKEN saved to file.")
        except Exception as e:
            logging.warning(f"Could not save token: {e}")
