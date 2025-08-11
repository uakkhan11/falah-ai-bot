# order_tracker.py

from datetime import datetime
import logging

class OrderTracker:
    def __init__(self, kite, trade_logger):
        self.kite = kite
        self.trade_logger = trade_logger
        self.logger = logging.getLogger(__name__)

    def update_order_statuses(self):
        """Fetch orders and update trade log with execution details."""
        try:
            orders = self.kite.orders()
            for o in orders:
                symbol = o["tradingsymbol"]
                status = o["status"]
                filled_qty = o["filled_quantity"]
                avg_price = o["average_price"]
                order_id = o["order_id"]

                # Update in Google Sheet / CSV
                self.trade_logger.log_trade(
                    symbol, 
                    "BUY" if o["transaction_type"] == "BUY" else "SELL", 
                    filled_qty, avg_price, 
                    status
                )
        except Exception as e:
            self.logger.error(f"Failed to update order statuses: {e}")

    def get_positions_with_pl(self):
        """Fetch current positions with P/L."""
        try:
            positions = self.kite.positions()["net"]
            result = []
            for p in positions:
                pl = p["unrealised"] + p["realised"]
                result.append({
                    "symbol": p["tradingsymbol"],
                    "qty": p["quantity"],
                    "avg_price": p["average_price"],
                    "last_price": p["last_price"],
                    "pnl": pl
                })
            return result
        except Exception as e:
            self.logger.error(f"Failed to fetch positions: {e}")
            return []
