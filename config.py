# config.py

import json
import os
import logging
from kiteconnect import KiteConnect

TOKENS_FILE = "kite_tokens.json"

class Config:
    def __init__(self):
        # Trading parameters...
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

        # Zerodha API credentials
        self.API_KEY = "your_api_key_here"
        self.API_SECRET = "your_api_secret_here"
        self.ACCESS_TOKEN = None
        self.REQUEST_TOKEN = None

        # Telegram credentials
        self.TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
        self.TELEGRAM_CHAT_ID = "your_telegram_chat_id"

        # KiteConnect client placeholder
        self.kite = None

        logging.basicConfig(level=logging.INFO)

    def _load_tokens(self):
        """Load saved tokens from disk if available."""
        if os.path.exists(TOKENS_FILE):
            try:
                with open(TOKENS_FILE, "r") as f:
                    data = json.load(f)
                    self.ACCESS_TOKEN = data.get("access_token")
                    logging.info("Loaded ACCESS_TOKEN from disk.")
            except Exception as e:
                logging.warning(f"Failed to load tokens file: {e}")

    def _save_tokens(self):
        """Save ACCESS_TOKEN to disk."""
        try:
            with open(TOKENS_FILE, "w") as f:
                json.dump({"access_token": self.ACCESS_TOKEN}, f)
                logging.info("ACCESS_TOKEN saved to disk.")
        except Exception as e:
            logging.warning(f"Failed to save tokens file: {e}")

    def authenticate(self):
        """Initialize and authenticate KiteConnect, persisting tokens."""
        # Attempt to load existing token
        self._load_tokens()

        # Initialize client
        self.kite = KiteConnect(api_key=self.API_KEY)

        # If we have a token, use it
        if self.ACCESS_TOKEN:
            try:
                self.kite.set_access_token(self.ACCESS_TOKEN)
                logging.info("Authenticated with persisted ACCESS_TOKEN.")
                return
            except Exception:
                logging.warning("Persisted ACCESS_TOKEN invalid, re-authenticating.")

        # No valid token, start interactive login
        login_url = self.kite.login_url()
        print(f"\n🔑 Login URL: {login_url}\n")
        print("• Open the URL, login & complete 2FA")
        print("• You will be redirected to your redirect URI with a request_token")
        self.REQUEST_TOKEN = input("Enter the request_token from redirect URL: ").strip()

        # Exchange for access token
        try:
            session_data = self.kite.generate_session(
                self.REQUEST_TOKEN, api_secret=self.API_SECRET
            )
            self.ACCESS_TOKEN = session_data["access_token"]
            self.kite.set_access_token(self.ACCESS_TOKEN)
            logging.info("Authentication successful, ACCESS_TOKEN obtained.")
            self._save_tokens()
        except Exception as e:
            logging.error(f"Failed to generate session: {e}")
            raise
