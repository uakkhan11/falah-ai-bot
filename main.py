#!/usr/bin/env python3
import sys
import signal
import logging
import threading
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI
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

app = FastAPI()

def update_analysis_data():
    try:
        logging.info("📊 Updating historical data & indicators before strategy execution...")
        fetcher = SmartHalalFetcher()
        fetcher.fetch_all()
        logging.info("✅ Data update completed successfully.")
    except Exception as e:
        logging.error(f"❌ Data update failed: {e}")

class FalahTradingBot:
    def __init__(self):
        self.config = Config()
        self.kite = self.config.kite
        self.running = False
        import threading as th
        if th.current_thread() is th.main_thread():
            signal.signal(signal.SIGINT, self.shutdown)
            signal.signal(signal.SIGTERM, self.shutdown)
        self.config.authenticate()
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
        self.last_status = {}
        self.last_summary_date = None
        self.daily_trade_count = 0
        self.current_batch_size = 25
        self.min_batch_size = 5
        self.max_batch_size = 25

        # Load instruments and trading list, add check for None
        self.data_manager.get_instruments()
        if not hasattr(self.data_manager, 'instruments') or self.data_manager.instruments is None:
            logging.error("Error: data_manager.instruments is None after get_instruments()")
            self.instruments = {}
        else:
            self.instruments = self.data_manager.instruments

        self.trading_symbols = self.load_trading_symbols()
        self.instrument_tokens = [self.instruments[s] for s in self.trading_symbols if s in self.instruments]
        self.live_price_streamer = LivePriceStreamer(self.kite, self.instrument_tokens)

    def shutdown(self, signum, frame):
        print("\n🛑 Shutting down bot...")
        self.running = False

    def load_trading_symbols(self):
        if self.gsheet is None:
            fallback = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            print("⚠️ Using fallback symbols:", fallback)
            return fallback
        try:
            syms = self.gsheet.get_symbols_from_sheet(worksheet_name="HalalList")
        except Exception as e:
            print(f"Error fetching sheet symbols: {e}")
            syms = None
        if not syms:
            fallback = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            print("⚠️ Using fallback symbols:", fallback)
            return fallback
        print(f"📊 Trading {len(syms)} symbols")
        return syms

    # ... (rest of your class methods unchanged except removing backslashes)

    def run(self):
        print("🚀 Bot started")
        self.running = True
        if self.live_price_streamer._is_market_open():
            self.live_price_streamer.start()
        else:
            print("Market closed; skipping live price streaming.")
        while self.running:
            self.capital_manager.update_funds()
            self.execute_strategy()
            self.order_tracker.update_order_statuses()
            positions = self.order_tracker.get_positions_with_pl()
            positions_with_age = self.holding_tracker.get_holdings_with_age(positions)

            # Use asyncio.run for async methods but catch RuntimeError if loop running
            try:
                asyncio.run(self.notifier.send_pnl_update(positions_with_age))
            except RuntimeError:
                pass

            for pos in positions_with_age:
                if self.last_status.get(pos['symbol']) != pos['holding_status']:
                    try:
                        asyncio.run(self.notifier.send_t1_t2_change(pos['symbol'], pos['holding_status']))
                    except RuntimeError:
                        pass
                    self.last_status[pos['symbol']] = pos['holding_status']

            today = date.today()
            if self.last_summary_date != today:
                total_pnl = sum(p['pnl'] for p in positions_with_age)
                cap_summary = self.capital_manager.get_capital_summary()
                hold_lines = [
                    f"{p['symbol']}: Qty={p['qty']} | PnL=₹{p['pnl']:.2f} | Status={p['holding_status']}"
                    for p in positions_with_age
                ]
                summary_msg = (
                    f"📅 <b>Daily Summary ({today.strftime('%d-%m-%Y')})\n"
                    f"Total P&L: ₹{total_pnl:.2f}\n"
                    f"Trades Today: {self.daily_trade_count}\n\n"
                    f"<b>Capital:</b>\n"
                    f"Available: ₹{cap_summary['available']:,.0f}\n"
                    f"Utilization: {cap_summary['utilization_pct']:.1f}%\n\n"
                    f"<b>Holdings:</b>\n" +
                    ("\n".join(hold_lines) if hold_lines else "No holdings")
                )
                try:
                    asyncio.run(self.notifier.send_message(summary_msg))
                except RuntimeError:
                    pass
                self.last_summary_date = today
                self.daily_trade_count = 0

            self.exit_manager.check_and_exit_positions(positions)
            time.sleep(60)
        self.live_price_streamer.stop()

# Define bot and entry points
bot = None

def run_bot():
    global bot
    bot = FalahTradingBot()
    bot.run()

@app.on_event("startup")
def startup_event():
    threading.Thread(target=run_bot, daemon=True).start()

@app.get("/api/portfolio")
def get_portfolio():
    if bot:
        return {
            "portfolio_value": 25421,
            "todays_profit": "+7.2%",
            "open_trades": 12,
        }
    return {}

@app.get("/api/trades")
def get_trades():
    if bot:
        return [
            {"id": 1, "symbol": "AAPL", "quantity": 10, "price": 192.38, "status": "Open"},
            {"id": 2, "symbol": "TSLA", "quantity": 5, "price": 247.11, "status": "Closed"},
        ]
    return []

if __name__ == "__main__":
    update_analysis_data()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
