#!/usr/bin/env python3
import sys
import signal
import time
from datetime import date
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
        # Config
        self.config = Config()
        self.running = False
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        self.config.authenticate()

        # Core modules
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

        # State trackers
        self.last_status = {}
        self.last_summary_date = None
        self.daily_trade_count = 0
        self.current_batch_size = 25
        self.min_batch_size = 5
        self.max_batch_size = 25

        # Load instruments & symbols
        self.data_manager.get_instruments()
        self.trading_symbols = self.load_trading_symbols()

    def shutdown(self, signum, frame):
        print("\n🛑 Shutting down bot...")
        self.running = False

    def load_trading_symbols(self):
        syms = self.gsheet.get_symbols_from_sheet(worksheet_name="HalalList")
        if not syms:
            fallback = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            print("⚠️ Using fallback symbols:", fallback)
            return fallback
        print(f"📊 Trading {len(syms)} symbols")
        return syms

    def run(self):
        print("🚀 Bot started")
        self.running = True
        while self.running:
            self.execute_strategy()

            # Monitoring & alerts
            self.order_tracker.update_order_statuses()
            positions = self.order_tracker.get_positions_with_pl()
            positions_with_age = self.holding_tracker.get_holdings_with_age(positions)

            self.notifier.send_pnl_update(positions_with_age)

            # Settlement changes
            for pos in positions_with_age:
                if self.last_status.get(pos['symbol']) != pos['holding_status']:
                    self.notifier.send_t1_t2_change(pos['symbol'], pos['holding_status'])
                    self.last_status[pos['symbol']] = pos['holding_status']

            # Daily summary
            today = date.today()
            if self.last_summary_date != today:
                total_pnl = sum(p['pnl'] for p in positions_with_age)
                hold_lines = [
                    f"{p['symbol']}: Qty={p['qty']} | PnL=₹{p['pnl']:.2f} | Status={p['holding_status']}"
                    for p in positions_with_age
                ]
                summary_msg = (
                    f"📅 <b>Daily Summary ({today.strftime('%d-%m-%Y')})</b>\n"
                    f"Total P&L: ₹{total_pnl:.2f}\n"
                    f"Trades Today: {self.daily_trade_count}\n\n"
                    f"<b>Holdings:</b>\n" +
                    ("\n".join(hold_lines) if hold_lines else "No holdings")
                )
                self.notifier.send_message(summary_msg)
                self.last_summary_date = today
                self.daily_trade_count = 0

            time.sleep(60)

    def execute_strategy(self):
        symbols = self.trading_symbols

        for i in range(0, len(symbols), self.current_batch_size):
            batch = symbols[i:i + self.current_batch_size]

            # Fetch data with retry/backoff
            daily_data   = self.data_manager.get_historical_data_parallel(batch, interval="day", days=200)
            hourly_data  = self.data_manager.get_historical_data_parallel(batch, interval="60minute", days=60)
            fifteen_data = self.data_manager.get_historical_data_parallel(batch, interval="15minute", days=20)
            live_prices  = self.data_manager.get_bulk_current_prices(batch)

            # Adaptive batch adjustment
            if self.data_manager.rate_limit_hit:
                old_size = self.current_batch_size
                self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 5)
                print(f"⚠️ Rate limit detected. Reducing batch size {old_size} → {self.current_batch_size}")
            else:
                if self.current_batch_size < self.max_batch_size:
                    self.current_batch_size += 2
                    print(f"✅ Increasing batch size to {self.current_batch_size}")

            positions = self.order_tracker.get_positions_with_pl()
            pos_dict = {p['symbol']: p for p in positions}

            def process_symbol(symbol):
                try:
                    df_daily   = daily_data.get(symbol)
                    df_hourly  = hourly_data.get(symbol)
                    df_fifteen = fifteen_data.get(symbol)

                    if (df_daily is None or df_daily.empty or
                        df_fifteen is None or df_fifteen.empty):
                        return f"⚠️ Not enough data for {symbol}"

                    if symbol in pos_dict and pos_dict[symbol]['qty'] > 0:
                        return f"⏩ Already holding {symbol}"

                    # Trend
                    df_daily = self.add_indicators(df_daily)
                    daily_up = df_daily.iloc[-1]['close'] > df_daily.iloc[-1]['ema200']

                    hourly_ok = True
                    if df_hourly is not None and not df_hourly.empty:
                        df_hourly = self.add_indicators(df_hourly)
                        hourly_ok = df_hourly.iloc[-1]['close'] > df_hourly.iloc[-1]['ema200']

                    # Entry signals
                    df_fifteen = self.add_indicators(df_fifteen)
                    df_fifteen = self.breakout_signal(df_fifteen)
                    df_fifteen = self.bb_breakout_signal(df_fifteen)
                    df_fifteen = self.bb_pullback_signal(df_fifteen)
                    df_fifteen = self.combine_signals(df_fifteen)

                    latest = df_fifteen.iloc[-1]
                    if daily_up and hourly_ok and latest['entry_signal'] == 1:
                        price = live_prices.get(symbol)
                        if price and price > 0:
                            qty = int(self.config.POSITION_SIZE / price)
                            allowed, reason = self.risk_manager.allow_trade()
                            if qty > 0 and allowed:
                                order_id = self.order_manager.place_buy_order(symbol, qty, price=price)
                                if order_id:
                                    self.trade_logger.log_trade(symbol, "BUY", qty, price, "ORDER_PLACED")
                                    self.notifier.send_trade_alert(symbol, "BUY", qty, price, "ORDER_PLACED")
                                    self.daily_trade_count += 1
                                else:
                                    self.trade_logger.log_trade(symbol, "BUY", qty, price, "ORDER_FAILED")
                                    self.notifier.send_trade_alert(symbol, "BUY", qty, price, "ORDER_FAILED")
                                return f"✅ Order placed for {symbol}"
                            else:
                                self.notifier.send_message(f"⚠️ Trade blocked for {symbol}: {reason or 'Risk rules blocked'}")
                                return f"⏩ Blocked {symbol}: {reason}"
                    return f"ℹ️ No trade for {symbol}"

                except Exception as e:
                    return f"❌ Error processing {symbol}: {e}"

            with ThreadPoolExecutor(max_workers=10) as executor:
                for future in as_completed({executor.submit(process_symbol, s): s for s in batch}):
                    print(future.result())

    # Wrappers
    def add_indicators(self, df): return add_indicators(df)
    def breakout_signal(self, df): return breakout_signal(df)
    def bb_breakout_signal(self, df): return bb_breakout_signal(df)
    def bb_pullback_signal(self, df): return bb_pullback_signal(df)
    def combine_signals(self, df): return combine_signals(df)


if __name__ == "__main__":
    bot = FalahTradingBot()
    bot.run()
