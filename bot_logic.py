# bot_logic.py

import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from config import Config
from improved_fetcher import SmartHalalFetcher
from live_data_manager import LiveDataManager
from order_manager import OrderManager
from gsheet_manager import GSheetManager
from trade_logger import TradeLogger
from order_tracker import OrderTracker
from risk_manager import RiskManager
from holding_tracker import HoldingTracker
from telegram_notifier import TelegramNotifier
from exit_manager import ExitManager
from capital_manager import CapitalManager
from live_price_streamer import LivePriceStreamer
from live_candle_aggregator import LiveCandleAggregator
from strategy_utils import (add_indicators, breakout_signal, bb_breakout_signal,
                            bb_pullback_signal, combine_signals)


class FalahTradingBot:
    def __init__(self, kite, config):
        self.kite = kite
        self.config = config
        self.running = False

        self.data_manager = LiveDataManager(self.kite)
        self.order_manager = OrderManager(self.kite, self.config)

        try:
            self.gsheet = GSheetManager(
                credentials_file="falah-credentials.json",
                sheet_key="1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
            )
        except Exception as e:
            logging.error(f"Google Sheet setup failed: {e}")
            self.gsheet = None

        self.trade_logger = TradeLogger(
            csv_path="trade_log.csv",
            gsheet_manager=self.gsheet,
            gsheet_sheet_name="TradeLog"
        )
        self.order_tracker = OrderTracker(self.kite, self.trade_logger)
        self.holding_tracker = HoldingTracker("trade_log.csv")
        self.risk_manager = RiskManager(self.config, self.order_tracker)

        self.notifier = TelegramNotifier(
            bot_token=self.config.TELEGRAM_BOT_TOKEN,
            chat_id=self.config.TELEGRAM_CHAT_ID
        )
        self.capital_manager = CapitalManager(
            self.config, self.order_tracker, self.order_manager, self.notifier
        )
        self.exit_manager = ExitManager(
            self.config, self.data_manager, self.order_manager,
            self.trade_logger, self.notifier,
            state_file="exit_state.json"
        )

        # Instruments and symbols loading
        self.data_manager.get_instruments()
        self.instruments = getattr(self.data_manager, 'instruments', {}) or {}
        self.trading_symbols = self.load_trading_symbols()
        missing = [s for s in self.trading_symbols if s not in self.instruments]
        if missing:
            logging.error(f"Instrument token not found for: {', '.join(missing)}")
        self.instrument_tokens = [self.instruments[s] for s in self.trading_symbols if s in self.instruments]

        self.live_price_streamer = LivePriceStreamer(self.kite, self.instrument_tokens)
        self.live_candle_aggregator = LiveCandleAggregator(
            tokens=self.instrument_tokens,
            interval="15minute"
        )
        self.live_candle_aggregator.start()

        # Other initializations
        self.last_status = {}
        self.last_summary_date = None
        self.daily_trade_count = 0

    def load_trading_symbols(self):
        if self.gsheet is None:
            fallback = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            logging.warning("Using fallback symbols: %s", fallback)
            return fallback
        try:
            syms = self.gsheet.get_symbols_from_sheet(worksheet_name="HalalList")
        except Exception as e:
            logging.error(f"Error fetching sheet symbols: {e}")
            syms = None
        if not syms:
            fallback = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            logging.warning("Using fallback symbols: %s", fallback)
            return fallback
        return syms

    def run_cycle(self):
        """Run one iteration of update and strategy logic."""
        self.capital_manager.update_funds()
        live_candles = self.live_candle_aggregator.get_all_live_candles()
        self.execute_strategy(live_candles)
        self.order_tracker.update_order_statuses()
        # Add any additional periodic actions here

    def get_portfolio_summary(self):
        return {
            "portfolio_value": self.capital_manager.get_portfolio_value(),
            "todays_profit": self.capital_manager.get_today_profit(),
            "open_trades": len(self.order_tracker.get_positions_with_pl())
        }

    def get_positions(self):
        return self.order_tracker.get_positions_with_pl()

    # Your execute_strategy and other methods unchanged...

def create_bot_instance():
    config = Config()
    config.authenticate()
    if config.kite is None:
        raise RuntimeError("KiteConnect client not initialized.")
    return FalahTradingBot(config.kite, config)
