# config.py

import os
import json
import logging
from kiteconnect import KiteConnect

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

        # Zerodha API credentials
        self.API_KEY = "your_api_key_here"
        self.API_SECRET = "your_api_secret_here"

        # Telegram config (optional)
        self.TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
        self.TELEGRAM_CHAT_ID = "your_telegram_chat_id"

        self.ACCESS_TOKEN = None
        self.kite = None

        logging.basicConfig(level=logging.INFO)

    def _load_saved_token(self):
        """Load saved access token if present."""
        if os.path.exists(TOKENS_FILE):
            print(f"🔍 Loading access token from: {os.path.abspath(TOKENS_FILE)}")
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
        """Save current access token to file."""
        try:
            with open(TOKENS_FILE, "w") as f:
                json.dump({"access_token": self.ACCESS_TOKEN}, f)
            logging.info("ACCESS_TOKEN saved to file.")
        except Exception as e:
            logging.warning(f"Could not save token: {e}")

    def authenticate(self):
        self._load_saved_token()
        self.kite = KiteConnect(api_key=self.API_KEY)

    # Use saved token if valid
    if self.ACCESS_TOKEN:
        self.kite.set_access_token(self.ACCESS_TOKEN)
        try:
            user_profile = self.kite.profile()
            logging.info(f"Authenticated using saved ACCESS_TOKEN. User: {user_profile['user_name']}")
            print(f"Authenticated using saved ACCESS_TOKEN. User: {user_profile['user_name']}")
        return
    except Exception:
       logging.warning("Saved ACCESS_TOKEN invalid, login required.")

    # Manual new token flow
    login_url = self.kite.login_url()
    print(f"\n🔑 LOGIN URL:\n{login_url}\n")
    print("1. Open in browser & login with 2FA.")
    print("2. Copy the request_token from the redirected URL.")

    request_token = input("Paste request_token here: ").strip()

    try:
        session_data = self.kite.generate_session(request_token, api_secret=self.API_SECRET)
        print(f"Session data received: {session_data}")  # Debug print
        self.ACCESS_TOKEN = session_data.get("access_token")
        if not self.ACCESS_TOKEN:
            print("Error: Access token not found in session data.")
            return
        self.kite.set_access_token(self.ACCESS_TOKEN)
        self._save_token()
        logging.info("✅ Authentication successful and token saved.")
        print("✅ Authentication successful and token saved.")
    except Exception as e:
        logging.error(f"Authentication failed: {e}")
        print(f"Authentication failed: {e}")
        raise
