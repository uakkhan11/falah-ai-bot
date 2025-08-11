#!/usr/bin/env python3
import sys
import signal
import time
from datetime import datetime
import pandas as pd

from config import Config
from live_data_manager import LiveDataManager
from order_manager import OrderManager
from gsheet_manager import GSheetManager  # New import
from your_indicator_module import (
    add_indicators,
    breakout_signal,
    bb_breakout_signal,
    bb_pullback_signal,
    combine_signals
)

class FalahTradingBot:
    def __init__(self):
        self.config = Config()
        self.running = False
        self.setup_signal_handlers()
        
        # Initialize Google Sheets manager
        self.gsheet = GSheetManager('falah-credentials.json')
        
        # Your Google Sheet details
        self.SYMBOLS_SHEET_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"
        self.SYMBOLS_WORKSHEET = "Symbols"  # Name of your symbols worksheet
        
    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
    def shutdown(self, signum, frame):
        print("Shutting down bot...")
        self.running = False
        
    def load_trading_symbols(self):
        """Load symbols from your Google Sheet"""
        symbols = self.gsheet.get_symbols_from_sheet(
            self.SYMBOLS_SHEET_URL, 
            self.SYMBOLS_WORKSHEET
        )
        
        if not symbols:
            # Fallback to hardcoded symbols if sheet fails
            symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            print("⚠️  Using fallback symbols")
        
        print(f"📊 Trading {len(symbols)} symbols: {symbols[:10]}...")  # Show first 10
        return symbols
        
    def run(self):
        """Main bot loop"""
        try:
            # Authenticate with Zerodha
            self.config.authenticate()
            
            # Initialize components
            self.data_manager = LiveDataManager(self.config.kite)
            self.order_manager = OrderManager(self.config.kite, self.config)
            
            # Get instruments
            self.data_manager.get_instruments()
            
            # Load symbols from Google Sheet
            self.trading_symbols = self.load_trading_symbols()
            
            # Main trading loop
            self.running = True
            print("🚀 Falah Trading Bot started!")
            print(f"📈 Monitoring {len(self.trading_symbols)} symbols")
            
            while self.running:
                self.execute_strategy()
                time.sleep(60)  # Check every minute
                
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    def execute_strategy(self):
        """Execute your trading strategy on symbols from Google Sheet"""
        print(f"🔍 Scanning {len(self.trading_symbols)} symbols...")
        
        for symbol in self.trading_symbols:
            try:
                # Get historical data for indicators
                df = self.data_manager.get_historical_data(symbol)
                if df is None or df.empty:
                    continue
                
                # Apply your existing indicators and signals
                df = self.data_manager.get_historical_data(symbol)
                df = self.add_indicators(df)
                df = self.breakout_signal(df)
                df = self.bb_breakout_signal(df)
                df = self.bb_pullback_signal(df)
                df = self.combine_signals(df)
                
                # Check latest signal
                if len(df) > 0:
                    latest_signal = df.iloc[-1]['entry_signal']
                    signal_type = df.iloc[-1]['entry_type']
                    
                    if latest_signal == 1:
                        print(f"🎯 Signal detected for {symbol}: {signal_type}")
                        
                        # Calculate position size based on your existing logic
                        quantity = self.calculate_position_size(symbol, df.iloc[-1])
                        
                        if quantity > 0:
                            self.order_manager.place_buy_order(symbol, quantity)
                        
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
    
    def calculate_position_size(self, symbol, latest_data):
        """Calculate position size based on your existing logic"""
        try:
            current_price = self.data_manager.get_current_price(symbol)
            if not current_price:
                return 0
                
            # Use your existing position sizing logic
            position_value = self.config.POSITION_SIZE
            quantity = int(position_value / current_price)
            
            return quantity
            
        except Exception as e:
            print(f"Error calculating position size for {symbol}: {e}")
            return 0
    
    # Copy all your existing functions here:
    def add_indicators(self, df):
        # Your existing function
        pass
        
    def breakout_signal(self, df):
        # Your existing function
        pass
        
    # ... other functions

if __name__ == "__main__":
    bot = FalahTradingBot()
    bot.run()
