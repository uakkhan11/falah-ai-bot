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
from live_candle_aggregator import LiveCandleAggregator  # Import your new class
from your_indicators_module import add_indicators, breakout_signal, bb_breakout_signal, bb_pullback_signal, combine_signals  # Import these properly

app = FastAPI()

config = None
bot = None


def update_analysis_data():
    try:
        logging.info("📊 Updating historical data & indicators before strategy execution...")
        fetcher = SmartHalalFetcher()
        fetcher.fetch_all()
        logging.info("✅ Data update completed successfully.")
    except Exception as e:
        logging.error(f"❌ Data update failed: {e}")


class FalahTradingBot:
    def __init__(self, kite, config):
        self.kite = kite
        self.config = config
        self.running = False
        import threading as th
        if th.current_thread() is th.main_thread():
            # Remove signal handlers here; handle shutdown via FastAPI events instead to avoid threading issues
            pass

        # You already authenticated in startup, no need to re-authenticate here
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

        # Fetch instruments safely
        self.data_manager.get_instruments()
        if hasattr(self.data_manager, "instruments") and self.data_manager.instruments:
            self.instruments = self.data_manager.instruments
        else:
            logging.error("Error: Could not fetch instruments. Check API credentials, endpoints, and file paths.")
            self.instruments = {}

        # Load trading symbols once
        self.trading_symbols = self.load_trading_symbols()

        # Check and log missing instruments
        missing = [s for s in self.trading_symbols if s not in self.instruments]
        if missing:
            logging.error(f"Instrument token not found for: {', '.join(missing)}")

        # Safely create instrument tokens list
        self.instrument_tokens = [self.instruments[s] for s in self.trading_symbols if s in self.instruments]

        # Initialize live price streamer
        self.live_price_streamer = LivePriceStreamer(self.kite, self.instrument_tokens)

        # Initialize Live Candle Aggregator to get ongoing candle data
        self.live_candle_aggregator = LiveCandleAggregator(
            api_key=self.config.API_KEY,
            access_token=self.config.ACCESS_TOKEN,
            tokens=self.instrument_tokens,
            interval="15minute"  # Or "1hour", or customize as needed
        )
        self.live_candle_aggregator.start()

    def shutdown(self, signum, frame):
        print("\n🛑 Shutting down bot...")
        self.running = False
        self.live_candle_aggregator.stop()
        self.live_price_streamer.stop()

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

    def calculate_dynamic_position_size(self, symbol, price, atr):
        if atr is None or atr <= 0:
            return 0
        stop_loss_distance = atr * self.config.ATR_SL_MULT
        if stop_loss_distance <= 0 or stop_loss_distance > price * 0.5:
            return 0
        account_value = self.config.INITIAL_CAPITAL
        risk_amount = account_value * self.config.RISK_PER_TRADE
        qty = int(risk_amount / stop_loss_distance)
        if qty <= 0 or (qty * price) > account_value:
            return 0
        return qty

    def run(self):
        print("🚀 Bot started")
        self.running = True
        if self.live_price_streamer._is_market_open():
            self.live_price_streamer.start()
        else:
            print("Market closed; skipping live price streaming.")

        while self.running:
            self.capital_manager.update_funds()

            # Use pre-fetched historical data loaded once at startup (avoid repeat API calls)
            # Load from CSV or data_manager cache instead of calling get_historical_data_parallel every loop

            # Example: read daily data from CSV only once at startup or on new candle close
            # You must implement CSV reading logic according to your setup here if needed

            # Use live candle aggregator for current 15min candle data:
            live_candles = self.live_candle_aggregator.get_all_live_candles()

            # Modify your execute_strategy to use live_candles and cached historical data

            self.execute_strategy(live_candles)

            self.order_tracker.update_order_statuses()
            positions = self.order_tracker.get_positions_with_pl()
            positions_with_age = self.holding_tracker.get_holdings_with_age(positions)

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

    def execute_strategy(self, live_candles):
        symbols = self.trading_symbols
        for i in range(0, len(symbols), self.current_batch_size):
            batch = symbols[i:i + self.current_batch_size]

            # IMPORTANT: Do NOT fetch historical data here every loop to avoid rate limits!
            # Instead, load data once and cache it or read from CSV files.
            # Example (pseudo code):
            # daily_data = load_csv_data(batch, timeframe='day')
            # hourly_data = load_csv_data(batch, timeframe='60minute')
            # fifteen_data = load_csv_data(batch, timeframe='15minute')

            # Use live_candles from websocket aggregator instead of fetching current candle from API
            positions = self.order_tracker.get_positions_with_pl()
            pos_dict = {p['symbol']: p for p in positions}

            def process_symbol(symbol):
                try:
                    # Replace these with your cached historical data loaded once or from CSV files
                    df_daily = ...  # load historical daily data for symbol
                    df_hourly = ...  # load historical hourly data for symbol
                    df_fifteen = ...  # load historical 15 minute data for symbol

                    # Inject live candle OHLCV into df_fifteen as latest candle if needed:
                    if symbol in self.instruments:
                        token = self.instruments[symbol]
                        live_candle = live_candles.get(token)
                        if live_candle:
                            # Append or replace last row of df_fifteen with live_candle data
                            # Example:
                            live_candle_df = pd.DataFrame([{
                                'date': live_candle['ts'],
                                'open': live_candle['open'],
                                'high': live_candle['high'],
                                'low': live_candle['low'],
                                'close': live_candle['close'],
                                'volume': live_candle['volume'],
                            }])
                            df_fifteen = df_fifteen.iloc[:-1].append(live_candle_df, ignore_index=True)

                    if (df_daily is None or df_daily.empty or df_fifteen is None or df_fifteen.empty):
                        return f"⚠️ Not enough data for {symbol}"
                    if symbol in pos_dict and pos_dict[symbol]['qty'] > 0:
                        return f"⏩ Already holding {symbol}"

                    # Use your properly imported indicator functions here
                    df_daily = add_indicators(df_daily)
                    daily_up = df_daily.iloc[-1]['close'] > df_daily.iloc[-1]['ema200']

                    hourly_ok = True
                    if df_hourly is not None and not df_hourly.empty:
                        df_hourly = add_indicators(df_hourly)
                        hourly_ok = df_hourly.iloc[-1]['close'] > df_hourly.iloc[-1]['ema200']

                    df_fifteen = add_indicators(df_fifteen)
                    df_fifteen = breakout_signal(df_fifteen)
                    df_fifteen = bb_breakout_signal(df_fifteen)
                    df_fifteen = bb_pullback_signal(df_fifteen)
                    df_fifteen = combine_signals(df_fifteen)

                    latest = df_fifteen.iloc[-1]
                    if daily_up and hourly_ok and latest['entry_signal'] == 1:
                        price = self.live_price_streamer.get_price(symbol)
                        if price is not None and price > 0:
                            atr = latest['atr']
                            desired_qty = self.calculate_dynamic_position_size(symbol, price, atr)
                            qty, cap_reason = self.capital_manager.adjust_quantity_for_capital(symbol, price, desired_qty)
                            allowed, risk_reason = self.risk_manager.allow_trade()

                            if qty > 0 and allowed:
                                order_id = self.order_manager.place_buy_order(symbol, qty, price=price)
                                if order_id:
                                    self.capital_manager.allocate_capital(qty * price)
                                    self.trade_logger.log_trade(symbol, "BUY", qty, price, "ORDER_PLACED")
                                    try:
                                        asyncio.run(self.notifier.send_trade_alert(symbol, "BUY", qty, price, "ORDER_PLACED"))
                                    except RuntimeError:
                                        pass
                                    if cap_reason and desired_qty != qty:
                                        try:
                                            asyncio.run(
                                                self.notifier.send_message(
                                                    f"💰 {symbol} size adjusted: {desired_qty} → {qty} due to capital limits"
                                                )
                                            )
                                        except RuntimeError:
                                            pass
                                    self.daily_trade_count += 1
                                else:
                                    self.trade_logger.log_trade(symbol, "BUY", qty, price, "ORDER_FAILED")
                                    try:
                                        asyncio.run(self.notifier.send_trade_alert(symbol, "BUY", qty, price, "ORDER_FAILED"))
                                    except RuntimeError:
                                        pass
                                return f"✅ Order placed for {symbol} qty={qty}"
                            elif not allowed:
                                try:
                                    asyncio.run(self.notifier.send_message(f"⚠️ Trade blocked for {symbol}: {risk_reason}"))
                                except RuntimeError:
                                    pass
                                return f"⏩ Risk blocked {symbol}: {risk_reason}"
                            else:
                                try:
                                    asyncio.run(self.notifier.send_message(f"💰 Trade blocked for {symbol}: {cap_reason}"))
                                except RuntimeError:
                                    pass
                                return f"⏩ Capital blocked {symbol}: insufficient funds"
                    return f"ℹ️ No trade for {symbol}"
                except Exception as e:
                    return f"❌ Error processing {symbol}: {e}"

            with ThreadPoolExecutor(max_workers=10) as executor:
                for future in as_completed({executor.submit(process_symbol, s): s for s in batch}):
                    print(future.result())

bot = None

def run_bot():
    global bot
    bot.run()

@app.on_event("startup")
def startup_event():
    global config, bot
    config = Config()
    try:
        config.authenticate()
    except Exception as e:
        logging.error(f"Authentication error: {e}")
        sys.exit(1)
    if config.kite is None:
        logging.error("Error: KiteConnect client not initialized. Exiting.")
        sys.exit(1)
    bot = FalahTradingBot(config.kite, config)
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
