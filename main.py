#!/usr/bin/env python3
import sys
import signal
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from strategy_utils import (
    add_indicators,
    breakout_signal,
    bb_breakout_signal,
    bb_pullback_signal,
    combine_signals
)
from config import Config
from live_data_manager import LiveDataManager
from order_manager import OrderManager
from gsheet_manager import GSheetManager
from trade_logger import TradeLogger
from order_tracker import OrderTracker
from risk_manager import RiskManager
from holding_tracker import HoldingTracker
from telegram_notifier import TelegramNotifier


class FalahTradingBot:
    def __init__(self):
        # Config / auth
        self.config = Config()
        self.running = False
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        self.config.authenticate()

        # Core components
        self.data_manager = LiveDataManager(self.config.kite)
        self.order_manager = OrderManager(self.config.kite, self.config)
        self.gsheet = GSheetManager(
            credentials_file="falah-credentials.json",
            sheet_key="1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
        )
        self.trade_logger = TradeLogger(
            csv_path="trade_log.csv",
            gsheet_manager=self.gsheet,
            gsheet_sheet_name="TradeLog"
        )
        self.order_tracker = OrderTracker(self.config.kite, self.trade_logger)
        self.holding_tracker = HoldingTracker("trade_log.csv")
        self.risk_manager = RiskManager(self.config, self.order_tracker)
        self.notifier = TelegramNotifier(
            bot_token=self.config.TELEGRAM_BOT_TOKEN,
            chat_id=self.config.TELEGRAM_CHAT_ID
        )

        self.last_status = {}  # for T1/T2 change tracking

        # Load instrument tokens and trading symbols
        self.data_manager.get_instruments()
        self.trading_symbols = self.load_trading_symbols()

    def shutdown(self, signum, frame):
        print("\n🛑 Shutting down bot...")
        self.running = False

    def load_trading_symbols(self):
        syms = self.gsheet.get_symbols_from_sheet(
            worksheet_name="HalalList"
        )
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

            # 🔍 Monitoring block
            self.order_tracker.update_order_statuses()
            positions = self.order_tracker.get_positions_with_pl()
            positions_with_age = self.holding_tracker.get_holdings_with_age(positions)

            # Send Telegram P&L update
            self.notifier.send_pnl_update(positions_with_age)

            # Send settlement status change alerts
            for pos in positions_with_age:
                symbol = pos['symbol']
                status = pos['holding_status']
                prev_status = self.last_status.get(symbol)
                if prev_status != status:
                    self.notifier.send_t1_t2_change(symbol, status)
                    self.last_status[symbol] = status

            time.sleep(60)

    def execute_strategy(self):
        live_prices = self.data_manager.get_bulk_current_prices(self.trading_symbols)
        positions = self.order_tracker.get_positions_with_pl()
        positions_dict = {pos['symbol']: pos for pos in positions}

        def process_symbol(symbol):
            try:
                # Fetch multi-timeframe data
                df_daily = self.data_manager.get_historical_data(symbol, "day")
                df_hourly = self.data_manager.get_historical_data(symbol, "60minute")
                df_fifteen = self.data_manager.get_historical_data(symbol, "15minute")

                if (df_daily is None or df_daily.empty or
                        df_fifteen is None or df_fifteen.empty):
                    return f"⚠️ Not enough data for {symbol}, skipped."

                # Don't enter if position already open
                if symbol in positions_dict and positions_dict[symbol]['qty'] > 0:
                    return f"⏩ Position already open for {symbol}, skipping order."

                # Indicator logic
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
                        if qty > 0 and self.risk_manager.allow_trade():
                            order_id = self.order_manager.place_buy_order(symbol, qty, price=price)
                            if order_id:
                                self.trade_logger.log_trade(symbol, "BUY", qty, price, status="ORDER_PLACED")
                                self.notifier.send_trade_alert(symbol, "BUY", qty, price, "ORDER_PLACED")
                            else:
                                self.trade_logger.log_trade(symbol, "BUY", qty, price, status="ORDER_FAILED")
                                self.notifier.send_trade_alert(symbol, "BUY", qty, price, "ORDER_FAILED")
                            return f"✅ Order attempt for {symbol} qty={qty}"
                        else:
                            return f"⏩ Trade blocked by risk rules for {symbol}"
                return f"ℹ️ No trade for {symbol}"

            except Exception as e:
                return f"❌ Error processing {symbol}: {e}"

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(process_symbol, sym): sym for sym in self.trading_symbols}
            for future in as_completed(futures):
                print(future.result())

    # Wrappers
    def add_indicators(self, df): return add_indicators(df)
    def breakout_signal(self, df): return breakout_signal(df)
    def bb_breakout_signal(self, df): return bb_breakout_signal(df)
    def bb_pullback_signal(self, df): return bb_pullback_signal(df)
    def combine_signals(self, df): return combine_signals(df)

    def calculate_position_size(self, symbol, latest):
        try:
            price = self.data_manager.get_current_price(symbol)
            if not price or price <= 0:
                print(f"⚠️ No valid price for {symbol}, skipping position size calc.")
                return 0
            return max(int(self.config.POSITION_SIZE / price), 0)
        except Exception as e:
            print(f"Error calculating position size for {symbol}: {e}")
            return 0


if __name__ == "__main__":
    bot = FalahTradingBot()
    bot.run()
