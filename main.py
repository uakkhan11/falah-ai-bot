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
from concurrent.futures import ThreadPoolExecutor, as_completed
from trade_logger import TradeLogger
from order_tracker import OrderTracker

class FalahTradingBot:
    def __init__(self):
        # Existing init ...
        self.gsheet = GSheetManager(
            credentials_file="falah-credentials.json",
            sheet_key="1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"  # your sheet key
        )
        
        self.trade_logger = TradeLogger(
            csv_path="trade_log.csv",
            gsheet_manager=self.gsheet,
            gsheet_sheet_name="TradeLog"
        )
        self.order_tracker = OrderTracker(self.config.kite, self.trade_logger)
        
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
        live_prices = self.data_manager.get_bulk_current_prices(self.trading_symbols)
        
        def process_symbol(symbol):
            try:
                # Fetch multi-timeframe data
                df_daily   = self.data_manager.get_historical_data(symbol, "day")
                df_hourly  = self.data_manager.get_historical_data(symbol, "60minute")
                df_fifteen = self.data_manager.get_historical_data(symbol, "15minute")
    
                # Skip incomplete data
                if (df_daily is None or df_daily.empty or
                    df_fifteen is None or df_fifteen.empty):
                    return f"⚠️ Not enough data for {symbol}, skipped."
    
                # Apply indicator logic
                df_daily = self.add_indicators(df_daily)
                daily_trend_up = df_daily.iloc[-1]['close'] > df_daily.iloc[-1]['ema200']
    
                hourly_confirm = True
                if df_hourly is not None and not df_hourly.empty:
                    df_hourly = self.add_indicators(df_hourly)
                    hourly_confirm = df_hourly.iloc[-1]['close'] > df_hourly.iloc[-1]['ema200']
    
                df_fifteen = self.add_indicators(df_fifteen)
                df_fifteen = self.breakout_signal(df_fifteen)
                df_fifteen = self.bb_breakout_signal(df_fifteen)
                df_fifteen = self.bb_pullback_signal(df_fifteen)
                df_fifteen = self.combine_signals(df_fifteen)
    
                latest_15m = df_fifteen.iloc[-1]
    
                if daily_trend_up and hourly_confirm and latest_15m['entry_signal'] == 1:
                    price = live_prices.get(symbol)
                    if price and price > 0:
                        qty = int(self.config.POSITION_SIZE / price)
                        if qty > 0:
                            order_id = self.order_manager.place_buy_order(symbol, qty, price=price)
                
                            # ✅ Trade logging here
                            if order_id:
                                self.trade_logger.log_trade(symbol, "BUY", qty, price, status="ORDER_PLACED")
                            else:
                                self.trade_logger.log_trade(symbol, "BUY", qty, price, status="ORDER_FAILED")
                            return f"✅ Order placed for {symbol} qty={qty}"
                return f"ℹ️ No trade for {symbol}"
            except Exception as e:
                return f"❌ Error processing {symbol}: {e}"
    
        # Run in parallel threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in self.trading_symbols}
            for future in as_completed(futures):
                print(future.result())

    self.order_tracker.update_order_statuses()
        positions = self.order_tracker.get_positions_with_pl()
        for pos in positions:
            print(f"{pos['symbol']}: Qty={pos['qty']}, PnL={pos['pnl']:.2f}")

        # 3. Wait before next cycle
        time.sleep(60)

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

self.order_tracker.update_order_statuses()
positions = self.order_tracker.get_positions_with_pl()
for pos in positions:
    print(f"{pos['symbol']}: Qty={pos['qty']}, PnL={pos['pnl']:.2f}")

if __name__ == "__main__":
    bot = FalahTradingBot()
    bot.run()
