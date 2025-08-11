# main.py
#!/usr/bin/env python3
from strategy_utils import (
    add_indicators,
    breakout_signal,
    bb_breakout_signal,
    bb_pullback_signal,
    combine_signals
)
import sys
import signal
import time
from datetime import datetime

from config import Config
from live_data_manager import LiveDataManager
from order_manager import OrderManager
from gsheet_manager import GSheetManager

class FalahTradingBot:
    SHEET_KEY = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
    WORKSHEET = "HalalList"

    def __init__(self):
        self.config = Config()
        self.running = False
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        # authenticate once
        self.config.authenticate()
        # init components
        self.data_manager = LiveDataManager(self.config.kite)
        self.order_manager = OrderManager(self.config.kite, self.config)
        self.gsheet = GSheetManager()
        # load instruments
        self.data_manager.get_instruments()
        self.trading_symbols = self.load_trading_symbols()

    def shutdown(self, signum, frame):
        print("\n🛑 Shutting down bot...")
        self.running = False

    def load_trading_symbols(self):
        syms = self.gsheet.get_symbols_from_sheet(self.SHEET_KEY, self.WORKSHEET)
        if not syms:
            fallback = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            print("⚠️  Using fallback symbols:", fallback)
            return fallback
        print(f"📊 Trading {len(syms)} symbols")
        return syms

    def run(self):
        print("🚀 Falah Trading Bot started!")
        self.running = True
        while self.running:
            self.execute_strategy()
            time.sleep(60)

    def execute_strategy(self):
        for symbol in self.trading_symbols:
            try:
                df = self.data_manager.get_historical_data(symbol)
                if df is None or df.empty:
                    continue
                df = self.add_indicators(df)
                df = self.breakout_signal(df)
                df = self.bb_breakout_signal(df)
                df = self.bb_pullback_signal(df)
                df = self.combine_signals(df)
                latest = df.iloc[-1]
                if latest['entry_signal'] == 1:
                    qty = self.calculate_position_size(symbol, latest)
                    if qty > 0:
                        self.order_manager.place_buy_order(symbol, qty)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")

    # Wrap existing functions
    def add_indicators(self, df):       return add_indicators(df)
    def breakout_signal(self, df):     return breakout_signal(df)
    def bb_breakout_signal(self, df):  return bb_breakout_signal(df)
    def bb_pullback_signal(self, df):  return bb_pullback_signal(df)
    def combine_signals(self, df):      return combine_signals(df)

    def calculate_position_size(self, symbol, latest):
        try:
            # Get live price from LiveDataManager
            price = self.data_manager.get_current_price(symbol)
            if not price or price <= 0:
                print(f"⚠️ No valid price received for {symbol}, skipping position size calc.")
                return 0
    
            # Fixed rupee value per trade divided by current market price
            quantity = int(self.config.POSITION_SIZE / price)
            return max(quantity, 0)
    
        except Exception as e:
            print(f"Error calculating position size for {symbol}: {e}")
            return 0

if __name__ == "__main__":
    bot = FalahTradingBot()
    bot.run()
