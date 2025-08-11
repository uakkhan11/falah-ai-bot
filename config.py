# config.py

import os
from kiteconnect import KiteConnect

class Config:
    def __init__(self):
        # Zerodha API credentials (set as environment variables)
        self.API_KEY = os.getenv('ijzeuwuylr3g0kug')
        self.API_SECRET = os.getenv('yy1wd2wn8r0wx4mus00vxllgss03nuqx')
        
        # Trading parameters from your backtest
        self.INITIAL_CAPITAL = 1_000_000
        self.POSITION_SIZE = 100_000
        self.MAX_POSITIONS = 5
        
        # Initialize KiteConnect
        self.kite = KiteConnect(api_key=self.API_KEY)
        
    def authenticate(self):
        # Manual authentication for now - we'll automate this later
        login_url = self.kite.login_url()
        print(f"Visit: {login_url}")
        request_token = input("Enter request token: ")
        
        data = self.kite.generate_session(request_token, self.API_SECRET)
        self.kite.set_access_token(data["access_token"])
        return True
