# config.py

import os
import json
import logging
from kiteconnect import KiteConnect
from auto_auth import auto_authenticate  # your local redirect server auth helper

TOKENS_FILE = "kite_tokens.json"

class Config:
    def __init__(self):
        # --- Trading parameters ---
        self.API_KEY = "your_api_key_here"
        self.API_SECRET = "your_api_secret_here"

        # Telegram config (optional)
        self.TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
        self.TELEGRAM_CHAT_ID = "your_telegram_chat_id"

        self.ACCESS_TOKEN = None
        self.kite = None

        logging.basicConfig(level=logging.INFO)

    def authenticate(self):
        """
        Use the local redirect server flow to authenticate and
        obtain KiteConnect instance.
        """
        kite = auto_authenticate()
        if not kite:
            raise Exception("Authentication failed.")
        self.kite = kite

    def _load_saved_token(self):
        """
        (Optional, not required with auto_auth.py approach)
        Load saved access token from file if needed.
        """
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
        """
        (Optional, used in other approach)
        Save access token to file.
        """
        try:
            with open(TOKENS_FILE, "w") as f:
                json.dump({"access_token": self.ACCESS_TOKEN}, f)
            logging.info("ACCESS_TOKEN saved to file.")
        except Exception as e:
            logging.warning(f"Could not save token: {e}")
