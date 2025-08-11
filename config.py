# config.py

import os
import json
from kiteconnect import KiteConnect

class Config:
    TOKEN_FILE = "falah_token.json"
    CREDS_FILE = "falah-credentials.json"

    def __init__(self):
        self.API_KEY = os.getenv('ZERODHA_API_KEY')
        self.API_SECRET = os.getenv('ZERODHA_API_SECRET')
        if not self.API_KEY or not self.API_SECRET:
            raise ValueError("Set ZERODHA_API_KEY and ZERODHA_API_SECRET in your VPS environment")
        self.kite = KiteConnect(api_key=self.API_KEY)

    def authenticate(self):
        # Try loading saved token
        token = None
        if os.path.exists(self.TOKEN_FILE):
            with open(self.TOKEN_FILE) as f:
                data = json.load(f)
                token = data.get("access_token")
        if token:
            try:
                self.kite.set_access_token(token)
                # Test call to verify token validity
                self.kite.profile()
                print("✅ Loaded saved access token")
                return
            except Exception:
                print("⚠️  Saved token invalid, re-authenticating")
        # Request new token
        print("👉  Obtain new request token by logging in:")
        print(self.kite.login_url())
        req_token = input("Enter request token: ").strip()
        sess = self.kite.generate_session(req_token, self.API_SECRET)
        access_token = sess["access_token"]
        self.kite.set_access_token(access_token)
        # Save token
        with open(self.TOKEN_FILE, "w") as f:
            json.dump({"access_token": access_token}, f)
        print("✅ New access token saved")
