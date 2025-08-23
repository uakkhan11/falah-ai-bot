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
        self.current_batch_size = 25

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

    def get_portfolio_summary(self):
          try:
              return {
                  "portfolio_value": self.capital_manager.get_portfolio_value(),
                  "todays_profit": self.capital_manager.get_today_profit(),
                  "open_trades": len(self.order_tracker.get_positions_with_pl())
              }
          except Exception as e:
              return {f"Error getting portfolio summary: {e}"}

    def get_positions(self):
        try:
            return self.order_tracker.get_positions_with_pl()
        except Exception as e:
            return [{"error": str(e)}]

    def run_cycle(self):
        try:
          """Run one iteration of update and strategy logic."""
          self.capital_manager.update_funds()
          live_candles = self.live_candle_aggregator.get_all_live_candles()
          self.execute_strategy(live_candles)
          self.order_tracker.update_order_statuses()
          return "Bot cycle executed successfully."
        except Exception as e:
          return f"Error running bot cycle: {e}"


    def execute_strategy(self, live_candles):
        symbols = self.trading_symbols
        for i in range(0, len(symbols), self.current_batch_size):
            batch = symbols[i:i + self.current_batch_size]
            positions = self.order_tracker.get_positions_with_pl()
            pos_dict = {p['symbol']: p for p in positions}

            def process_symbol(symbol):
                try:
                    try:
                        df_daily = pd.read_csv(f"swing_data/{symbol}.csv")
                        df_daily['date'] = pd.to_datetime(df_daily['date'])
                        df_daily = df_daily.sort_values('date').reset_index(drop=True)
                        print(f"{symbol}: swing_data/{symbol}.csv loaded, shape={df_daily.shape}, min={df_daily['date'].min()}, max={df_daily['date'].max()}")
                    except Exception as e:
                        print(f"{symbol}: error loading swing_data/{symbol}.csv: {e}")
                        df_daily = None

                    try:
                        df_hourly = pd.read_csv(f"intraday_swing_data/{symbol}.csv")
                        df_hourly['date'] = pd.to_datetime(df_hourly['date'])
                        df_hourly = df_hourly.sort_values('date').reset_index(drop=True)
                        print(f"{symbol}: intraday_swing_data/{symbol}.csv loaded, shape={df_hourly.shape}, min={df_hourly['date'].min()}, max={df_hourly['date'].max()}")
                    except Exception as e:
                        print(f"{symbol}: error loading intraday_swing_data/{symbol}.csv: {e}")
                        df_hourly = None

                    try:
                        df_fifteen = pd.read_csv(f"scalping_data/{symbol}.csv")
                        df_fifteen['date'] = pd.to_datetime(df_fifteen['date'])
                        df_fifteen = df_fifteen.sort_values('date').reset_index(drop=True)
                        print(f"{symbol}: scalping_data/{symbol}.csv loaded, shape={df_fifteen.shape}, min={df_fifteen['date'].min()}, max={df_fifteen['date'].max()}")
                    except Exception as e:
                        print(f"{symbol}: error loading scalping_data/{symbol}.csv: {e}")
                        df_fifteen = None

                    for df, min_rows, label in [
                        (df_daily, 50, 'daily'), (df_fifteen, 20, '15m')
                    ]:
                        if df is not None and (df.empty or len(df) < min_rows):
                            print(f"{symbol}: {label} data insufficient ({len(df) if df is not None else 0}) rows")
                            if label == 'daily':
                                df_daily = None
                            if label == '15m':
                                df_fifteen = None

                    if df_daily is None or df_fifteen is None:
                        return f"⚠️ Not enough data for {symbol} (initial check)"

                    if symbol in self.instruments:
                        token = self.instruments[symbol]
                        live_candle = live_candles.get(token)
                        if live_candle:
                            live_candle_df = pd.DataFrame([{
                                'date': live_candle['ts'],
                                'open': live_candle['open'],
                                'high': live_candle['high'],
                                'low': live_candle['low'],
                                'close': live_candle['close'],
                                'volume': live_candle['volume'],
                            }])
                            if len(df_fifteen) > 0:
                                df_fifteen = pd.concat([df_fifteen.iloc[:-1], live_candle_df], ignore_index=True)
                            else:
                                df_fifteen = live_candle_df

                    print(f"{symbol} df_daily shape: {df_daily.shape if df_daily is not None else 'None'}")
                    print(f"{symbol} df_fifteen shape after live candle injection: {df_fifteen.shape if df_fifteen is not None else 'None'}")

                    if df_daily.empty or df_fifteen.empty:
                        return f"⚠️ Not enough data for {symbol} (after live candle injection)"

                    if symbol in pos_dict and pos_dict[symbol]['qty'] > 0:
                        return f"⏩ Already holding {symbol}"

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
                  

def create_bot_instance():
    config = Config()
    config.authenticate()
    if config.kite is None:
        raise RuntimeError("KiteConnect client not initialized.")
    return FalahTradingBot(config.kite, config)
