# risk_manager.py

from datetime import datetime
import logging

class RiskManager:
    def __init__(self, config, order_tracker):
        self.config = config
        self.order_tracker = order_tracker
        self.logger = logging.getLogger(__name__)
        self.daily_loss_limit = -self.config.DAILY_LOSS_LIMIT_PCT * self.config.INITIAL_CAPITAL
 
    def can_open_new_position(self):
        positions = self.order_tracker.get_positions_with_pl()
        open_positions = [p for p in positions if p['qty'] != 0]
        if len(open_positions) >= self.config.MAX_POSITIONS:
            return False, f"Max positions limit reached ({self.config.MAX_POSITIONS})"
        return True, None

    def check_daily_loss_limit(self):
        positions = self.order_tracker.get_positions_with_pl()
        total_pnl = sum(p['pnl'] for p in positions)
        if total_pnl <= self.daily_loss_limit:
            return False, f"Daily loss limit reached ({total_pnl:.2f})"
        return True, None

    def allow_trade(self):
        # First check positions
        ok, reason = self.can_open_new_position()
        if not ok:
            return False, reason
        # Then check P&L
        ok, reason = self.check_daily_loss_limit()
        if not ok:
            return False, reason
        return True, None

    def check_position_stop_loss(self, symbol, entry_price, current_price):
        drop_pct = (current_price - entry_price) / entry_price
        if drop_pct <= -self.config.MAX_POSITION_LOSS_PCT:
            self.logger.warning(f"🔻 Stop-loss triggered for {symbol} ({drop_pct*100:.2f}%).")
            return False
        return True

    def allow_trade(self):
        """ Master check for whether bot is allowed to place a new trade now. """
        return self.can_open_new_position() and self.check_daily_loss_limit()
