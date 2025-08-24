import json
import os

class ExitManager:
    def __init__(self, config, data_manager, order_manager, trade_logger, notifier, state_file="exit_state.json"):
        self.state_file = state_file
        self.positions = {}  # symbol -> {entry_price, shares, stop_loss_price, ...}
        self.config = config
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.trade_logger = trade_logger
        self.notifier = notifier
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                try:
                    self.positions = json.load(f)
                except json.JSONDecodeError:
                    self.positions = {}

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.positions, f, indent=2)

    def update_position(self, symbol, position_data):
        """Update or add position state and persist immediately."""
        self.positions[symbol] = position_data
        self.save_state()

    def remove_position(self, symbol):
        """Remove position from state and persist."""
        if symbol in self.positions:
            del self.positions[symbol]
            self.save_state()

    def get_position(self, symbol):
        """Get stored position state by symbol."""
        return self.positions.get(symbol)

    def update_trailing_stop(self, symbol, latest_row):
        """
        Update trailing stop loss price for the symbol based on latest candle indicators.
        Returns updated stop price or None if no position.
        """
        pos = self.get_position(symbol)
        if not pos:
            return None

        entry_price = pos.get('entry_price')
        current_sl = pos.get('stop_loss_price', entry_price - self.config.ATR_MULT * latest_row['atr'])
        chand_sl = latest_row.get('chandelier_exit', 0)
        atr_sl = entry_price - self.config.ATR_MULT * latest_row['atr']
        new_sl = max(chand_sl, atr_sl)
        updated_sl = max(current_sl, new_sl)

        if updated_sl != current_sl:
            pos['stop_loss_price'] = updated_sl
            self.update_position(symbol, pos)

        return updated_sl
