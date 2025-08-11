# exit_manager.py
from datetime import datetime
import logging
import pandas as pd


class ExitManager:
    def __init__(self, config, data_manager, order_manager, trade_logger, notifier):
        """
        Handles live trade exit logic using backtest rules.
        """
        self.config = config
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.trade_logger = trade_logger
        self.notifier = notifier
        self.logger = logging.getLogger(__name__)

        # Tracks consecutive regime fails to match backtest behaviour
        self.regime_fail_count = {}
        # Tracks trailing stop state for each symbol
        self.trailing_state = {}

    def check_and_exit_positions(self, positions):
        """
        For each open position, apply exit rules:
        1. Profit target
        2. ATR stop loss
        3. Chandelier Exit trailing stop
        4. Regime fail exit after 2 bars
        """
        for pos in positions:
            symbol = pos['symbol']
            qty = pos['qty']

            # Skip non-open positions
            if qty <= 0:
                continue

            # Fetch recent candles for exit timeframe (15min works for intraday updates)
            df = self.data_manager.get_historical_data(symbol, interval="15minute", days=5)
            if df is None or df.empty:
                continue

            # Add indicators from strategy_utils (same as backtest)
            from strategy_utils import add_indicators
            df = add_indicators(df)
            latest = df.iloc[-1]

            entry_price = pos['avg_price']
            current_price = pos['last_price']
            ret = (current_price - entry_price) / entry_price
            atr_stop = entry_price - self.config.ATR_SL_MULT * latest['atr']

            # State tracking key
            key = symbol

            # Initialise trailing state if first time
            if key not in self.trailing_state:
                self.trailing_state[key] = {
                    'high_since': entry_price,
                    'trail_active': False,
                    'trail_stop': 0
                }

            # Update high price since entry
            if current_price > self.trailing_state[key]['high_since']:
                self.trailing_state[key]['high_since'] = current_price

            # Activate trailing stop if trigger met
            if not self.trailing_state[key]['trail_active'] and ret >= self.config.TRAIL_TRIGGER:
                self.trailing_state[key]['trail_active'] = True
                self.trailing_state[key]['trail_stop'] = latest['chandelier_exit']

            # Ratchet up trailing stop if chandelier exit rises
            if (self.trailing_state[key]['trail_active'] and
                    latest['chandelier_exit'] > self.trailing_state[key]['trail_stop']):
                self.trailing_state[key]['trail_stop'] = latest['chandelier_exit']

            # Regime check (price above EMA200 AND ADX strong)
            regime_ok = (latest['close'] > latest['ema200']) and (latest['adx'] > self.config.ADX_THRESHOLD_DEFAULT)
            if not regime_ok:
                self.regime_fail_count[key] = self.regime_fail_count.get(key, 0) + 1
            else:
                self.regime_fail_count[key] = 0

            # Decide exit reason based on priority
            reason = None
            if ret >= self.config.PROFIT_TARGET:
                reason = 'Profit Target'
            elif current_price <= atr_stop:
                reason = 'ATR Stop Loss'
            elif (self.trailing_state[key]['trail_active'] and
                  current_price <= self.trailing_state[key]['trail_stop']):
                reason = 'Chandelier Exit'
            elif self.regime_fail_count.get(key, 0) >= 2:
                reason = 'Regime Exit'

            # If exit triggered, send SELL order
            if reason:
                try:
                    order_id = self.order_manager.place_sell_order(symbol, qty, price=current_price)
                    self.trade_logger.log_trade(symbol, "SELL", qty, current_price,
                                                status="EXIT_" + reason.replace(" ", "_"))
                    self.notifier.send_message(
                        f"🚪 Exit {symbol} ({reason}) @ ₹{current_price:.2f}"
                    )
                    self.logger.info(f"Exited {symbol} — Reason: {reason}, Qty: {qty}, Price: {current_price}")
                except Exception as e:
                    self.logger.error(f"Exit order failed for {symbol}: {e}")
