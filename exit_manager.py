import json
import os

class ExitManager:
    def __init__(self, config, data_manager, order_manager, trade_logger, notifier, state_file="exit_state.json"):
        self.state_file = state_file
        self.positions = {}  # symbol -> {entry_price, shares, stop_loss_price, ...}
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
        self.positions[symbol] = position_data
        self.save_state()

    def remove_position(self, symbol):
        if symbol in self.positions:
            del self.positions[symbol]
            self.save_state()

    def get_position(self, symbol):
        return self.positions.get(symbol)
