# exit_manager.py
from datetime import datetime
import logging
import json
import os


class ExitManager:
    def __init__(self, config, data_manager, order_manager, trade_logger, notifier,
                 state_file="exit_state.json"):
        """
        Handles live trade exit logic using backtest rules with persistent trailing state,
        plus Telegram alerts when stops tighten.
        """
        self.config = config
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.trade_logger = trade_logger
        self.notifier = notifier
        self.logger = logging.getLogger(__name__)

        self.state_file = state_file
        self.regime_fail_count = {}
        self.trailing_state = {}

        # Load previous state if available
        self._load_state()

    def _load_state(self):
        """Load trailing stop and regime fail states from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.trailing_state = data.get("trailing_state", {})
                    self.regime_fail_count = data.get("regime_fail_count", {})
                print(f"💾 ExitManager state loaded from {self.state_file}")
            except Exception as e:
                print(f"⚠️ Failed to load exit state: {e}")

    def _save_state(self):
        """Save trailing stop and regime fail states to file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump({
                    "trailing_state": self.trailing_state,
                    "regime_fail_count": self.regime_fail_count
                }, f)
        except Exception as e:
            print(f"⚠️ Failed to save exit state: {e}")

    def check_and_exit_positions(self, positions):
        """
        For each live position, check exit conditions and place sell orders if triggered.
        Sends Telegram alerts if trailing stop is ratcheted upward.
        """
        for pos in positions:
            symbol = pos['symbol']
            qty = pos['qty']
            if qty <= 0:
                continue

            # Pull recent candle data for exit logic
            df = self.data_manager.get_historical_data(symbol, interval="15minute", days=5)
            if df is None or df.empty:
                continue

            from strategy_utils import add_indicators
            df = add_indicators(df)
            latest = df.iloc[-1]

            entry_price = pos['avg_price']
            current_price = pos['last_price']
            ret = (current_price - entry_price) / entry_price
            atr_stop = entry_price - self.config.ATR_SL_MULT * latest['atr']

            # Init trailing state if not present
            if symbol not in self.trailing_state:
                self.trailing_state[symbol] = {
                    'high_since': entry_price,
                    'trail_active': False,
                    'trail_stop': 0
                }

            # Update high since entry
            if current_price > self.trailing_state[symbol]['high_since']:
                self.trailing_state[symbol]['high_since'] = current_price

            # Activate trailing if trigger hit
            if not self.trailing_state[symbol]['trail_active'] and ret >= self.config.TRAIL_TRIGGER:
                self.trailing_state[symbol]['trail_active'] = True
                self.trailing_state[symbol]['trail_stop'] = latest['chandelier_exit']
                self.notifier.send_message(
                    f"🔔 Trailing stop ACTIVATED for {symbol} at ₹{self.trailing_state[symbol]['trail_stop']:.2f}"
                )

            # Ratchet trailing stop upwards & notify
            if (self.trailing_state[symbol]['trail_active'] and
                    latest['chandelier_exit'] > self.trailing_state[symbol]['trail_stop']):
                old_trail_stop = self.trailing_state[symbol]['trail_stop']
                self.trailing_state[symbol]['trail_stop'] = latest['chandelier_exit']
                self.notifier.send_message(
                    f"🔔 Trailing stop RATCHET for {symbol}: "
                    f"{old_trail_stop:.2f} → {self.trailing_state[symbol]['trail_stop']:.2f}"
                )

            # Regime check
            regime_ok = (latest['close'] > latest['ema200']) and (latest['adx'] > self.config.ADX_THRESHOLD_DEFAULT)
            if not regime_ok:
                self.regime_fail_count[symbol] = self.regime_fail_count.get(symbol, 0) + 1
            else:
                self.regime_fail_count[symbol] = 0

            # Decide exit reason
            reason = None
            if ret >= self.config.PROFIT_TARGET:
                reason = 'Profit Target'
            elif current_price <= atr_stop:
                reason = 'ATR Stop Loss'
            elif (self.trailing_state[symbol]['trail_active'] and
                  current_price <= self.trailing_state[symbol]['trail_stop']):
                reason = 'Chandelier Exit'
            elif self.regime_fail_count.get(symbol, 0) >= 2:
                reason = 'Regime Exit'

            # Execute exit if triggered
            if reason:
                try:
                    self.order_manager.place_sell_order(symbol, qty, price=current_price)
                    self.trade_logger.log_trade(symbol, "SELL", qty, current_price,
                                                status="EXIT_" + reason.replace(" ", "_"))
                    self.notifier.send_message(
                        f"🚪 Exit {symbol} ({reason}) @ ₹{current_price:.2f}"
                    )
                    self.logger.info(f"Exited {symbol} — Reason: {reason}, Qty: {qty}, Price: {current_price}")

                    # Remove from state after exit
                    self.trailing_state.pop(symbol, None)
                    self.regime_fail_count.pop(symbol, None)

                except Exception as e:
                    self.logger.error(f"Exit order failed for {symbol}: {e}")

        # Save after processing all symbols
        self._save_state()
