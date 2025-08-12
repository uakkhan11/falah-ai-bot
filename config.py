# config.py

from kiteconnect import KiteConnect
import logging

class Config:
    def __init__(self):
        # Trading Parameters
        self.MAX_POSITION_LOSS_PCT = 0.03   # 3% per-position stop-loss
        self.INITIAL_CAPITAL = 1_000_000
        self.POSITION_SIZE = 100_000
        self.RISK_PER_TRADE = 0.01          # 1% account risk per trade
        self.PROFIT_TARGET = 0.10           # 10% profit target
        self.ATR_SL_MULT = 2.8              # ATR multiplier for stop-loss
        self.ATR_PERIOD = 14
        self.TRAIL_TRIGGER = 0.07           # 7% return to activate trailing
        self.TRAIL_DISTANCE = 0.03
        self.ADX_THRESHOLD_DEFAULT = 15
        self.MAX_POSITIONS = 5
        self.DAILY_LOSS_LIMIT_PCT = 0.05    # 5% daily loss limit
        
        # Zerodha API credentials
        self.API_KEY = "your_api_key_here"
        self.API_SECRET = "your_api_secret_here"
        self.ACCESS_TOKEN = None
        self.REQUEST_TOKEN = None
        
        # Telegram credentials
        self.TELEGRAM_BOT_TOKEN = "your_telegram_bot_token"
        self.TELEGRAM_CHAT_ID = "your_telegram_chat_id"
        
        # KiteConnect instance
        self.kite = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
    
    def authenticate(self):
        """Initialize and authenticate KiteConnect"""
        try:
            self.kite = KiteConnect(api_key=self.API_KEY)
            
            # If you have a stored access token, use it
            if self.ACCESS_TOKEN:
                self.kite.set_access_token(self.ACCESS_TOKEN)
            else:
                # Generate login URL (you'll need to handle login flow)
                login_url = self.kite.login_url()
                print(f"Login URL: {login_url}")
                # After login, you'll get request_token to generate access_token
                # self.ACCESS_TOKEN = self.kite.generate_session(
                #     self.REQUEST_TOKEN, api_secret=self.API_SECRET
                # )["access_token"]
                
        except Exception as e:
            print(f"Authentication failed: {e}")
            raise
