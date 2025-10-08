# config.py

import os
import json
import logging
import webbrowser
from kiteconnect import KiteConnect
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse

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
        
        # Zerodha API (replace with your keys)
        self.API_KEY = "ijzeuwuylr3g0kug"
        self.API_SECRET = "yy1wd2wn8r0wx4mus00vxllgss03nuqx"
        self.ACCESS_TOKEN = None
        self.kite = None
        
        # Telegram (optional)
        self.TELEGRAM_BOT_TOKEN = "7763450358:AAEghRYX0b8yvxq4V9nWKeyGlCiLwv1Oiz0"
        self.TELEGRAM_CHAT_ID = "6784139148"

        logging.basicConfig(level=logging.INFO)
        print(f"📍 Token file path set to: {os.path.abspath(TOKENS_FILE)}")

    def _load_saved_token(self):
        if os.path.exists(TOKENS_FILE):
            print(f"🔍 Loading access token from: {os.path.abspath(TOKENS_FILE)}")
            try:
                with open(TOKENS_FILE, "r") as f:
                    data = json.load(f)
                    token = data.get("access_token")
                    if token:
                        self.ACCESS_TOKEN = token
                        logging.info("Loaded ACCESS_TOKEN from file.")
                    else:
                        logging.warning("Token file found but no access_token key present.")
            except Exception as e:
                logging.warning(f"Could not load saved token: {e}")
        else:
            print(f"⚠️ No token file found at {os.path.abspath(TOKENS_FILE)}")

    def _save_token(self):
        try:
            with open(TOKENS_FILE, "w") as f:
                json.dump({"access_token": self.ACCESS_TOKEN}, f)
            logging.info(f"ACCESS_TOKEN saved to: {os.path.abspath(TOKENS_FILE)}")
        except Exception as e:
            logging.warning(f"Could not save token: {e}")

    def authenticate(self, request_token=None):
        """Authenticate either by saved token or by using request_token."""
        self._load_saved_token()
        self.kite = KiteConnect(api_key=self.API_KEY)

        if request_token is None:
            if self.ACCESS_TOKEN:
                print("🔑 Trying saved ACCESS_TOKEN...")
                self.kite.set_access_token(self.ACCESS_TOKEN)
                try:
                    profile = self.kite.profile()
                    logging.info(f"Authenticated using saved ACCESS_TOKEN. User: {profile['user_name']}")
                    print(f"✅ Authenticated as {profile['user_name']}")
                    return
                except Exception:
                    logging.warning("🚫 Saved ACCESS_TOKEN invalid, need fresh login.")
            # No valid token - require manual login
            login_url = self.kite.login_url()
            print(f"\n🔗 LOGIN URL:\n{login_url}\n")
            print("1️⃣ Open in browser & login with 2FA.")
            print("2️⃣ Copy the request_token from the redirected URL.")
            request_token = input("Paste request_token here: ").strip()

        try:
            print("Exchanging request_token for access_token ...")
            session_data = self.kite.generate_session(request_token, api_secret=self.API_SECRET)
            self.ACCESS_TOKEN = session_data.get("access_token")
            if not self.ACCESS_TOKEN:
                print("❌ No access_token received in session data")
                return
            self.kite.set_access_token(self.ACCESS_TOKEN)
            self._save_token()
            logging.info("✅ Authentication successful and token saved.")
            print("✅ Authentication successful and token saved.")
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            raise
