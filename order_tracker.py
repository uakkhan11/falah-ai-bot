# order_tracker.py

from datetime import datetime
import logging

class OrderTracker:
    def __init__(self, kite, trade_logger):
        self.kite = kite
        self.trade_logger = trade_logger
        self.logger = logging.getLogger(__name__)

    def update_order_statuses(self):
        """
        Fetch all orders from Kite Connect and log their status,
        filled quantities, and average prices back into the trade log.
        """
        try:
            orders = self.kite.orders()
            for o in orders:
                symbol   = o["tradingsymbol"]
                action   = o["transaction_type"]
                status   = o["status"]
                filled   = o.get("filled_quantity", 0)
                avg_price= o.get("average_price", 0.0)

                # Log each update (status can be: COMPLETE, CANCELLED, REJECTED, etc.)
                self.trade_logger.log_trade(
                    symbol,
                    action,
                    filled,
                    avg_price,
                    status
                )
        except Exception as e:
            self.logger.error(f"Failed to update order statuses: {e}")

    def get_positions_with_pl(self):
        """
        Fetch current net positions and calculate P/L for each.
        Returns a list of dicts: [{symbol, qty, avg_price, last_price, pnl}, …]
        """
        try:
            net_positions = self.kite.positions().get("net", [])
            result = []
            for p in net_positions:
                symbol     = p["tradingsymbol"]
                qty        = p.get("quantity", 0)
                avg_price  = p.get("average_price", 0.0)
                last_price = p.get("last_price", 0.0)
                realised   = p.get("realised", 0.0)
                unrealised = p.get("unrealised", 0.0)
                pnl        = realised + unrealised

                result.append({
                    "symbol": symbol,
                    "qty": qty,
                    "avg_price": avg_price,
                    "last_price": last_price,
                    "pnl": pnl
                })
            return result

        except Exception as e:
            self.logger.error(f"Failed to fetch positions: {e}")
            return []
