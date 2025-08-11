# config.py

import os
import json
from kiteconnect import KiteConnect

class Config:
    TOKEN_FILE = "falah_token.json"
    CREDS_FILE = "falah-credentials.json"
    self.MAX_POSITION_LOSS_PCT = 0.03   # 3% per-position stop-loss
    self.DAILY_LOSS_LIMIT_PCT = 0.02    # 2% total daily loss before halt
    self.TELEGRAM_BOT_TOKEN = os.getenv("7763450358:AAH32bWYyu_hXR6l-UaVMaarFGZ4YFOv6q8")
    self.TELEGRAM_CHAT_ID   = os.getenv("6784139148")

    def __init__(self):
        self.API_KEY = os.getenv('ZERODHA_API_KEY')
        self.API_SECRET = os.getenv('ZERODHA_API_SECRET')
        if not self.API_KEY or not self.API_SECRET:
            raise ValueError("Set ZERODHA_API_KEY and ZERODHA_API_SECRET in your VPS environment")

        self.kite = KiteConnect(api_key=self.API_KEY)

        # Add trading parameters here as class attributes
        self.POSITION_SIZE = 100_000          # Position size per trade
        self.INITIAL_CAPITAL = 1_000_000      # Total capital (if used)
        self.RISK_PER_TRADE = 0.01          # Risk 1% of account capital per trade
        self.MAX_POSITIONS = 5                 # Max concurrent positions
        self.PROFIT_TARGET = 0.10              # Profit target as fraction
        self.ATR_SL_MULT = 2.8                 # ATR multiplier for stop loss
        self.ATR_PERIOD = 14                   # ATR calculation period
        self.TRAIL_TRIGGER = 0.07              # Trailing stop trigger
        self.TRAIL_DISTANCE = 0.03             # Trailing stop distance
        self.TRANSACTION_COST = 0.001          # Transaction cost fraction

    def authenticate(self):
        # Your existing token handling code here
        token = None
        if os.path.exists(self.TOKEN_FILE):
            with open(self.TOKEN_FILE) as f:
                data = json.load(f)
                token = data.get("access_token")
        if token:
            try:
                self.kite.set_access_token(token)
                self.kite.profile()  # test token
                print("✅ Loaded saved access token")
                return
            except Exception:
                print("⚠️  Saved token invalid, re-authenticating")
        print("👉  Obtain new request token by logging in:")
        print(self.kite.login_url())
        req_token = input("Enter request token: ").strip()
        sess = self.kite.generate_session(req_token, self.API_SECRET)
        access_token = sess["access_token"]
        self.kite.set_access_token(access_token)
        with open(self.TOKEN_FILE, "w") as f:
            json.dump({"access_token": access_token}, f)
        print("✅ New access token saved")
