# capital_manager.py
import logging

class CapitalManager:
    def __init__(self, config, order_tracker, order_manager, notifier):
        """
        Manages available capital/margin and prevents over-allocation.
        """
        self.config = config
        self.order_tracker = order_tracker
        self.order_manager = order_manager
        self.notifier = notifier
        self.logger = logging.getLogger(__name__)
        self.available_funds = 0
        self.allocated_capital = 0
        self.margin_buffer = 0.05  # Keep 5% buffer
        self.equity_history = []   # To track portfolio value history for drawdowns

    def update_funds(self):
        """
        Fetch available margin/funds from broker API and calculate allocated capital.
        Also update equity history.
        """
        try:
            # Fetch margin info from Zerodha
            margins = self.order_manager.kite.margins()
            if 'equity' in margins:
                self.available_funds = margins['equity'].get('available', {}).get('cash', 0)
            else:
                # Fallback to config capital
                self.available_funds = self.config.INITIAL_CAPITAL
        except Exception as e:
            self.logger.warning(f"Could not fetch margin info: {e}")
            # Use config as fallback
            self.available_funds = self.config.INITIAL_CAPITAL
        
        self._update_allocated_capital()
        # Update equity history
        portfolio_value = self.get_portfolio_value()
        if portfolio_value is not None:
            self.equity_history.append(portfolio_value)

    def _update_allocated_capital(self):
        """
        Calculate total capital currently allocated in open positions.
        """
        positions = self.order_tracker.get_positions_with_pl()
        total_allocated = 0
        for pos in positions:
            if pos['qty'] > 0:
                total_allocated += pos['avg_price'] * pos['qty']
        self.allocated_capital = total_allocated

    def get_available_capital(self):
        """
        Returns free capital available for new trades (with buffer).
        """
        free_capital = self.available_funds - self.allocated_capital
        buffered_capital = free_capital * (1 - self.margin_buffer)
        return max(0, buffered_capital)

    def can_allocate(self, required_capital):
        """
        Check if required capital can be allocated for a trade.
        Returns:
            tuple: (can_allocate: bool, available_capital: float, max_possible: float)
        """
        available = self.get_available_capital()
        can_trade = available >= required_capital
        return can_trade, available, available

    def adjust_quantity_for_capital(self, symbol, price, desired_qty):
        """
        Adjust quantity based on available capital constraints.
        Returns:
            tuple: (adjusted_qty: int, adjustment_reason: str or None)
        """
        required_capital = price * desired_qty
        can_trade, available, _ = self.can_allocate(required_capital)
        if can_trade:
            return desired_qty, None
        max_qty = int(available / price) if price > 0 else 0
        if max_qty <= 0:
            return 0, f"Insufficient capital: need ₹{required_capital:,.0f}, available ₹{available:,.0f}"
        else:
            return max_qty, f"Capital adjusted: need ₹{required_capital:,.0f}, available ₹{available:,.0f}"

    def allocate_capital(self, capital_amount):
        """
        Mark capital as allocated (call after successful order placement).
        """
        self.allocated_capital += capital_amount

    def free_capital(self, capital_amount):
        """
        Free allocated capital after closing positions.
        """
        self.allocated_capital = max(0, self.allocated_capital - capital_amount)

    def get_capital_summary(self):
        """
        Returns capital allocation summary for monitoring.
        """
        utilization_pct = (self.allocated_capital / self.available_funds * 100) if self.available_funds > 0 else 0
        return {
            'total_funds': self.available_funds,
            'allocated': self.allocated_capital,
            'available': self.get_available_capital(),
            'utilization_pct': utilization_pct
        }

    # --- Additions for dashboard and bot integrations ---

    def get_portfolio_value(self):
        """
        Returns the total portfolio value.
        Customize as needed to calculate real portfolio value.
        """
        try:
            positions = self.order_tracker.get_positions_with_pl()
            total_value = self.available_funds
            for pos in positions:
                total_value += pos['last_price'] * pos['qty']
            return total_value
        except Exception as e:
            self.logger.error(f"Error in get_portfolio_value: {e}")
            return self.available_funds

    def get_today_profit(self):
        """
        Returns today's profit (PnL).
        Customize with your actual calculation.
        """
        try:
            positions = self.order_tracker.get_positions_with_pl()
            pnl = sum(pos.get('pnl', 0) for pos in positions)
            return pnl
        except Exception as e:
            self.logger.error(f"Error in get_today_profit: {e}")
            return 0

    def get_equity_curve(self):
        """
        Returns equity history list required for cooling mode drawdown calculations.
        """
        return self.equity_history
