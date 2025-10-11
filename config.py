#!/usr/bin/env python3
# config.py

import os
import json
import logging
from datetime import datetime
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

        # Zerodha API (replace with your keys)
        self.API_KEY = "ijzeuwuylr3g0kug"
        self.API_SECRET = "yy1wd2wn8r0wx4mus00vxllgss03nuqx"
        self.ACCESS_TOKEN = None
        self.kite = None

        # Telegram (optional)
        self.TELEGRAM_BOT_TOKEN = "7763450358:AAEghRYX0b8yvxq4V9nWKeyGlCiLwv1Oiz0"
        self.TELEGRAM_CHAT_ID = "6784139148"

        # Google Sheets settings (service account)
        self.gs_service_account_json = "/root/falah-ai-bot/falah-credentials.json"
        self.gs_spreadsheet_id = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
        self.gs_worksheet_name = "Summary"

        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
        print(f"📍 Token file path set to: {os.path.abspath(TOKENS_FILE)}")

        # Try to load any saved token immediately
        self._load_saved_token()
        if self.ACCESS_TOKEN:
            # Prepare a KiteConnect instance using the saved token,
            # but defer profile verification until authenticate() is called.
            try:
                self.kite = KiteConnect(api_key=self.API_KEY)
                self.kite.set_access_token(self.ACCESS_TOKEN)
            except Exception:
                # Will be re-created in authenticate paths
                self.kite = None

    # ---------------- Token persistence ----------------

    def _load_saved_token(self):
        """Load ACCESS_TOKEN from TOKENS_FILE if present."""
        try:
            if os.path.exists(TOKENS_FILE):
                print(f"🔍 Loading access token from: {os.path.abspath(TOKENS_FILE)}")
                with open(TOKENS_FILE, "r") as f:
                    data = json.load(f)
                token = data.get("access_token")
                if token:
                    self.ACCESS_TOKEN = token
                    logging.info("Loaded ACCESS_TOKEN from file.")
                else:
                    logging.warning("Token file found but no 'access_token' key present.")
            else:
                print(f"⚠️ No token file found at {os.path.abspath(TOKENS_FILE)}")
        except Exception as e:
            logging.warning(f"Could not load saved token: {e}")

    def _save_token(self):
        """Persist ACCESS_TOKEN to TOKENS_FILE."""
        try:
            payload = {
                "access_token": self.ACCESS_TOKEN,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            with open(TOKENS_FILE, "w") as f:
                json.dump(payload, f)
            logging.info(f"ACCESS_TOKEN saved to: {os.path.abspath(TOKENS_FILE)}")
        except Exception as e:
            logging.warning(f"Could not save token: {e}")

    def get_login_url(self):
    kc = self.kite if self.kite else KiteConnect(api_key=self.API_KEY)
    try:
    url = kc.login_url()
    except Exception:
    kc = KiteConnect(api_key=self.API_KEY)
    url = kc.login_url()
    self.kite = kc
    return url

    # --------------- Authentication paths ---------------

    def authenticate_with_token(self, request_token: str):
        """
        Exchange a request_token (from Zerodha login redirect) for an access_token,
        persist it, and set it on self.kite. Returns the access_token on success.
        Intended for programmatic calls (e.g., Gradio Control Center).
        """
        if not request_token or not request_token.strip():
            raise ValueError("request_token is required")

        self.kite = KiteConnect(api_key=self.API_KEY)
        logging.info("Exchanging request_token for access_token ...")
        data = self.kite.generate_session(request_token.strip(), api_secret=self.API_SECRET)
        access_token = data.get("access_token")
        if not access_token:
            raise RuntimeError("No access_token received from generate_session()")

        self.ACCESS_TOKEN = access_token
        self.kite.set_access_token(access_token)
        self._save_token()

        try:
            profile = self.kite.profile()
            logging.info(f"✅ Authenticated as {profile.get('user_name','UNKNOWN')}")
        except Exception as e:
            logging.warning(f"Token set but profile fetch failed: {e}")

        return access_token

    def authenticate(self, request_token: str | None = None):
        """
        Human-friendly auth: if a valid saved token exists, use it.
        Otherwise, print the login URL and prompt for a request_token (if not provided),
        then exchange and save.
        """
        # If we already have a Kite instance with a token, validate it
        if self.kite and self.ACCESS_TOKEN:
            try:
                profile = self.kite.profile()
                logging.info(f"Authenticated using saved ACCESS_TOKEN. User: {profile.get('user_name','UNKNOWN')}")
                print(f"✅ Authenticated as {profile.get('user_name','UNKNOWN')}")
                return
            except Exception:
                logging.warning("🚫 Saved ACCESS_TOKEN invalid, need fresh login.")

        # Otherwise (re)create Kite instance
        self.kite = KiteConnect(api_key=self.API_KEY)

        if not request_token:
            # No token provided: guide the user to log in and paste request_token
            login_url = self.kite.login_url()
            print("\n🔗 LOGIN URL:\n" + login_url + "\n")
            print("1️⃣ Open in browser & complete login with 2FA.")
            print("2️⃣ Copy the request_token from the redirected URL (query param).")
            request_token = input("Paste request_token here: ").strip()

        # Exchange and persist
        try:
            logging.info("Exchanging request_token for access_token ...")
            data = self.kite.generate_session(request_token.strip(), api_secret=self.API_SECRET)
            access_token = data.get("access_token")
            if not access_token:
                print("❌ No access_token received in session data")
                return
            self.ACCESS_TOKEN = access_token
            self.kite.set_access_token(access_token)
            self._save_token()
            logging.info("✅ Authentication successful and token saved.")
            try:
                profile = self.kite.profile()
                print(f"✅ Authenticated as {profile.get('user_name','UNKNOWN')}")
            except Exception:
                print("✅ Token saved, profile fetch skipped.")
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            raise
