# exit_manager.py
from datetime import datetime
import logging
import json
import os


class ExitManager:
    def __init__(self, config, data_manager, order_manager, trade_logger, notifier,
                 state_file="exit_state.json"):
        self.config = config
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.trade_logger = trade_logger
        self.notifier = notifier
        self.logger = logging.getLogger(__name__)
        self.state_file = state_file

        # State vars
        self.regime_fail_count = {}
        self.trailing_state = {}

        # Load persisted state
        self._load_state()

    def _load_state(self):
        """Load trailing and regime states from JSON file."""
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
        """Save trailing and regime states to JSON file."""
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
        Apply exit logic to open positions.
        """
        for pos in positions:
            symbol = pos['symbol']
            qty = pos['qty']
            if qty <= 0:
                continue

            # Fresh data for exit logic
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

            # Initialise if not existing (loaded or new)
            if symbol not in self.trailing_state:
                self.trailing_state[symbol] = {
                    'high_since': entry_price,
                    'trail_active': False,
                    'trail_stop': 0
                }

            # Update high since entry
            if current_price > self.trailing_state[symbol]['high_since']:
                self.trailing_state[symbol]['high_since'] = current_price

            # Activate trail if trigger met
            if not self.trailing_state[symbol]['trail_active'] and ret >= self.config.TRAIL_TRIGGER:
                self.trailing_state[symbol]['trail_active'] = True
                self.trailing_state[symbol]['trail_stop'] = latest['chandelier_exit']

            # Ratchet trail stop upwards
            if (self.trailing_state[symbol]['trail_active'] and
                    latest['chandelier_exit'] > self.trailing_state[symbol]['trail_stop']):
                self.trailing_state[symbol]['trail_stop'] = latest['chandelier_exit']

            # Regime OK check
            regime_ok = (latest['close'] > latest['ema200']) and (latest['adx'] > self.config.ADX_THRESHOLD_DEFAULT)
            if not regime_ok:
                self.regime_fail_count[symbol] = self.regime_fail_count.get(symbol, 0) + 1
            else:
                self.regime_fail_count[symbol] = 0

            # Exit reason priority
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

            if reason:
                try:
                    order_id = self.order_manager.place_sell_order(symbol, qty, price=current_price)
                    self.trade_logger.log_trade(symbol, "SELL", qty, current_price,
                                                status="EXIT_" + reason.replace(" ", "_"))
                    self.notifier.send_message(
                        f"🚪 Exit {symbol} ({reason}) @ ₹{current_price:.2f}"
                    )
                    self.logger.info(f"Exited {symbol}: {reason}, qty={qty}, price={current_price}")
                    # Reset state after exit
                    if symbol in self.trailing_state:
                        del self.trailing_state[symbol]
                    if symbol in self.regime_fail_count:
                        del self.regime_fail_count[symbol]
                except Exception as e:
                    self.logger.error(f"Exit order failed for {symbol}: {e}")

        # Save state every run
        self._save_state()
