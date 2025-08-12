# config.py

import os
import json
import logging
from kiteconnect import KiteConnect

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

    def _load_saved_token(self):
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
        try:
            with open(TOKENS_FILE, "w") as f:
                json.dump({"access_token": self.ACCESS_TOKEN}, f)
            logging.info("ACCESS_TOKEN saved to file.")
        except Exception as e:
            logging.warning(f"Could not save token: {e}")

    def authenticate(self):
        """Authenticates KiteConnect with saved or fresh token."""
        self._load_saved_token()
        self.kite = KiteConnect(api_key=self.API_KEY)

        # If we already have a token, set it
        if self.ACCESS_TOKEN:
            self.kite.set_access_token(self.ACCESS_TOKEN)
            logging.info("Authenticated using saved token.")
            return

        # Otherwise, do interactive login flow
        login_url = self.kite.login_url()
        print(f"\n🔑 LOGIN URL:\n{login_url}\n")
        print("1. Open in browser & login with 2FA.")
        print("2. You'll be redirected to your app's redirect URL.")
        print("3. Copy the `request_token` from that URL.")

        request_token = input("Paste request_token here: ").strip()

        try:
            session_data = self.kite.generate_session(
                request_token, api_secret=self.API_SECRET
            )
            self.ACCESS_TOKEN = session_data["access_token"]
            self.kite.set_access_token(self.ACCESS_TOKEN)
            self._save_token()
            logging.info("✅ Authentication successful.")
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            raise
