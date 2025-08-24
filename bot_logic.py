import logging
import numpy as np
import pandas as pd
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
from strategy_utils import add_indicators  # Extend as needed

ATR_MULT = 2.0

class FalahTradingBot:
    def __init__(self):
        self.kite = None
        self.config = Config()
        self.authenticated = False
        self.data_manager = None
        self.order_manager = None
        self.gsheet = None
        self.trade_logger = None
        self.order_tracker = None
        self.holding_tracker = None
        self.risk_manager = None
        self.notifier = None
        self.capital_manager = None
        self.exit_manager = None
        self.live_price_streamer = None
        self.live_candle_aggregator = None
        self.trading_symbols = []
        self.instruments = {}
        self.instrument_tokens = []
        # Cooling variables
        self.cooling_mode = False
        self.max_drawdown_threshold = 10.0  # % drawdown to start cooling
        self.min_capital_to_trade = 10000   # minimal capital to allow trade

        try:
            self.config.authenticate()
            if self.config.kite and self.config.ACCESS_TOKEN:
                self.kite = self.config.kite
                self.authenticated = True
                self._post_auth_setup()
        except Exception as e:
            logging.error(f"Authentication failed on init: {e}")
            self.authenticated = False

    def _post_auth_setup(self):
        self.data_manager = LiveDataManager(self.kite)
        self.order_manager = OrderManager(self.kite, self.config)
        try:
            self.gsheet = GSheetManager(
                credentials_file="falah-credentials.json",
                sheet_key="1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
            )
        except Exception as e:
            logging.error(f"Failed to setup GSheetManager: {e}")
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
        self.data_manager.get_instruments()
        self.instruments = getattr(self.data_manager, 'instruments', {}) or {}
        self.trading_symbols = self.load_trading_symbols()
        missing = [s for s in self.trading_symbols if s not in self.instruments]
        if missing:
            logging.error(f"Tokens not found for symbols: {', '.join(missing)}")
        self.instrument_tokens = [self.instruments[s] for s in self.trading_symbols if s in self.instruments]
        self.live_price_streamer = LivePriceStreamer(self.kite, self.instrument_tokens)
        self.live_candle_aggregator = LiveCandleAggregator(
            tokens=self.instrument_tokens,
            interval="15minute"
        )
        self.live_candle_aggregator.start()

    def is_authenticated(self):
        return self.authenticated

    def authenticate_with_token(self, request_token):
        self.config.authenticate(request_token=request_token)
        self.kite = self.config.kite
        self.authenticated = True
        self._post_auth_setup()

    def load_trading_symbols(self):
        if self.gsheet is None:
            fallback = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            logging.warning(f"Using fallback symbols: {fallback}")
            return fallback
        try:
            syms = self.gsheet.get_symbols_from_sheet(worksheet_name="HalalList")
        except Exception as e:
            logging.error(f"Error fetching sheet symbols: {e}")
            syms = None
        if not syms:
            fallback = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
            logging.warning(f"Using fallback symbols: {fallback}")
            return fallback
        return syms

    def check_exit_conditions(self, position, live_row):
        symbol = position['symbol']
        current_stop_loss = self.exit_manager.update_trailing_stop(symbol, live_row)
        if current_stop_loss is None:
            entry_price = position['entry_price']
            current_stop_loss = max(
                live_row.get('chandelier_exit', 0),
                entry_price - ATR_MULT * live_row.get('atr', 0)
            )
            self.exit_manager.update_position(symbol, {
                'entry_price': entry_price,
                'qty': position['qty'],
                'stop_loss_price': current_stop_loss
            })
        stop_loss_hit = live_row['low'] <= current_stop_loss if 'low' in live_row else False
        rsi = live_row.get('rsi_14', 100)
        close_price = live_row['close']
        bb_upper = live_row.get('bb_upper', 0)
        supertrend_dir = live_row.get('supertrend_direction', 1)
        momentum_exit = (rsi < 70 and close_price < bb_upper) or (supertrend_dir < 0)
        return stop_loss_hit, momentum_exit, current_stop_loss

    def check_and_apply_cooling(self):
        equity_curve = self.capital_manager.get_equity_curve()  # Implement this method to return list of floats
        if not equity_curve or len(equity_curve) < 2:
            self.cooling_mode = False
            self.config.RISK_PER_TRADE = 0.01 * self.capital_manager.get_portfolio_value()
            return
        eq_arr = np.array(equity_curve)
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns = (running_max - eq_arr) / running_max * 100
        current_drawdown = np.max(drawdowns)
        if current_drawdown >= self.max_drawdown_threshold:
            if not self.cooling_mode:
                self.cooling_mode = True
                self.config.RISK_PER_TRADE = max(0.005 * self.capital_manager.get_portfolio_value(), 1000)
                try:
                    import asyncio
                    asyncio.run(self.notifier.send_message(f"⚠️ Cooling Mode Activated! Drawdown reached {current_drawdown:.2f}%"))
                except RuntimeError:
                    pass
        else:
            if self.cooling_mode and current_drawdown < self.max_drawdown_threshold * 0.8:
                self.cooling_mode = False
                self.config.RISK_PER_TRADE = 0.01 * self.capital_manager.get_portfolio_value()
                try:
                    import asyncio
                    asyncio.run(self.notifier.send_message("✅ Cooling Mode Deactivated! Drawdown recovered"))
                except RuntimeError:
                    pass

    def run_cycle(self):
        if not self.authenticated:
            return "Bot not authenticated yet."

        self.capital_manager.update_funds()
        self.check_and_apply_cooling()

        if self.capital_manager.get_available_capital() < self.min_capital_to_trade:
            return "⏩ Insufficient capital to place new trades."

        if self.cooling_mode:
            return "⏩ Trading paused due to cooling mode."

        live_candles = self.live_candle_aggregator.get_all_live_candles()
        executed = 0
        failed = 0
        blocked_risk = 0
        blocked_capital = 0
        results = []

        positions = self.order_tracker.get_positions_with_pl()
        pos_dict = {p['symbol']: p for p in positions} if positions else {}

        for symbol in self.trading_symbols:
            try:
                import pandas as pd
                try:
                    df_daily = pd.read_csv(f"swing_data/{symbol}.csv", parse_dates=['date'])
                    df_daily = df_daily.sort_values('date').reset_index(drop=True)
                except Exception as e:
                    results.append(f"{symbol}: Error loading daily CSV - {e}")
                    continue
                try:
                    df_15m = pd.read_csv(f"scalping_data/{symbol}.csv", parse_dates=['date'])
                    df_15m = df_15m.sort_values('date').reset_index(drop=True)
                except Exception as e:
                    results.append(f"{symbol}: Error loading 15m CSV - {e}")
                    continue
                if df_daily.empty or df_15m.empty or len(df_daily) < 50 or len(df_15m) < 20:
                    results.append(f"{symbol}: Not enough data")
                    continue

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
                        df_15m = pd.concat([df_15m.iloc[:-1], live_candle_df], ignore_index=True)

                df_daily = add_indicators(df_daily)
                df_15m = add_indicators(df_15m)

                if symbol in pos_dict and pos_dict[symbol]['qty'] > 0:
                    latest_row = df_15m.iloc[-1]
                    stop_loss_hit, momentum_exit, current_stop_loss = self.check_exit_conditions(pos_dict[symbol], latest_row)
                    
                    if stop_loss_hit or momentum_exit:
                        qty = pos_dict[symbol]['qty']
                        order_id = self.order_manager.place_sell_order(symbol, qty, price=latest_row['close'])
                        if order_id:
                            self.capital_manager.free_capital(qty * latest_row['close'])
                            self.trade_logger.log_trade(symbol, "SELL", qty, latest_row['close'], "ORDER_PLACED")
                            try:
                                import asyncio
                                asyncio.run(self.notifier.send_trade_alert(symbol, "SELL", qty, latest_row['close'], "ORDER_PLACED"))
                            except RuntimeError:
                                pass
                            results.append(f"{symbol} Sell order placed qty={qty}")
                            executed += 1
                            self.exit_manager.remove_position(symbol)
                        else:
                            self.trade_logger.log_trade(symbol, "SELL", qty, latest_row['close'], "ORDER_FAILED")
                            results.append(f"{symbol} Sell order failed")
                            failed += 1
                    else:
                        results.append(f"{symbol} Holding position, no exit signal")
                    continue

                latest_row = df_15m.iloc[-1]
                daily_up = df_daily.iloc[-1]['close'] > df_daily.iloc[-1].get('ema200', 0)

                if daily_up and latest_row.get('entry_signal', 0) == 1:
                    price = self.live_price_streamer.get_price(symbol)
                    if price is None or price <= 0:
                        results.append(f"{symbol}: Invalid live price")
                        continue
                    atr = latest_row.get('atr', None)
                    if atr is None or atr <= 0:
                        results.append(f"{symbol}: Invalid ATR")
                        continue
                    risk_per_share = ATR_MULT * atr
                    desired_qty = int(self.config.RISK_PER_TRADE / risk_per_share)
                    qty, cap_reason = self.capital_manager.adjust_quantity_for_capital(symbol, price, desired_qty)
                    allowed, risk_reason = self.risk_manager.allow_trade()

                    if qty > 0 and allowed:
                        order_id = self.order_manager.place_buy_order(symbol, qty, price=price)
                        if order_id:
                            self.capital_manager.allocate_capital(qty * price)
                            self.trade_logger.log_trade(symbol, "BUY", qty, price, "ORDER_PLACED")
                            try:
                                import asyncio
                                asyncio.run(self.notifier.send_trade_alert(symbol, "BUY", qty, price, "ORDER_PLACED"))
                            except RuntimeError:
                                pass
                            if cap_reason and desired_qty != qty:
                                try:
                                    asyncio.run(self.notifier.send_message(f"💰 {symbol} size adjusted: {desired_qty} -> {qty} due to capital limits"))
                                except RuntimeError:
                                    pass
                            executed += 1
                            results.append(f"{symbol}: Order placed qty={qty}")
                            # Initialize trailing stop loss state for exit manager
                            self.exit_manager.update_position(symbol, {
                                'entry_price': price,
                                'qty': qty,
                                'stop_loss_price': price - self.config.ATR_MULT * atr
                            })
                        else:
                            self.trade_logger.log_trade(symbol, "BUY", qty, price, "ORDER_FAILED")
                            failed += 1
                            results.append(f"{symbol}: Order failed")
                    elif not allowed:
                        blocked_risk += 1
                        try:
                            import asyncio
                            asyncio.run(self.notifier.send_message(f"⚠️ Trade blocked for {symbol}: {risk_reason}"))
                        except RuntimeError:
                            pass
                        results.append(f"⏩ Risk blocked {symbol}: {risk_reason}")
                    else:
                        blocked_capital += 1
                        try:
                            import asyncio
                            asyncio.run(self.notifier.send_message(f"💰 Trade blocked for {symbol}: {cap_reason}"))
                        except RuntimeError:
                            pass
                        results.append(f"⏩ Capital blocked {symbol}: insufficient funds")
                else:
                    results.append(f"{symbol}: Entry conditions not met or daily trend down")
            except Exception as e:
                results.append(f"❌ Error processing {symbol}: {e}")

        summary = (
            f"Total symbols processed: {len(self.trading_symbols)}\n"
            f"Orders executed: {executed}\n"
            f"Orders failed: {failed}\n"
            f"Blocked by risk: {blocked_risk}\n"
            f"Blocked by capital: {blocked_capital}\n"
            "Details:\n" + "\n".join(results)
        )
        return summary

def create_bot_instance():
    return FalahTradingBot()
