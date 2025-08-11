# holding_tracker.py

# holding_tracker.py
import csv
import os
from datetime import datetime
from dateutil import parser
import logging

class HoldingTracker:
    def __init__(self, trade_log_csv="trade_log.csv"):
        self.trade_log_csv = trade_log_csv
        self.logger = logging.getLogger(__name__)

    def get_holdings_with_age(self, positions):
        """
        Takes current positions from OrderTracker and augments with age/T1-T2 info.
        positions: list of dicts from OrderTracker.get_positions_with_pl()
        """
        purchase_dates = self._load_latest_buy_dates()
        today = datetime.now().date()
        result = []

        for pos in positions:
            symbol = pos['symbol']
            qty = pos['qty']

            if qty <= 0:
                continue  # skip shorts or closed
            buy_date = purchase_dates.get(symbol)
            if not buy_date:
                status = "Unknown"
                days = None
            else:
                days = (today - buy_date).days
                if days == 0:
                    status = "T1 (today's buy)"
                elif days == 1:
                    status = "T1"
                elif days == 2:
                    status = "T2"
                else:
                    status = "Settled"

            pos['holding_status'] = status
            pos['holding_days'] = days
            result.append(pos)

        return result

    def _load_latest_buy_dates(self):
        purchase_dates = {}
        if not os.path.exists(self.trade_log_csv):
            return purchase_dates

        try:
            with open(self.trade_log_csv, newline='') as f:
                reader = csv.DictReader(
                    f,
                    fieldnames=["timestamp", "symbol", "action", "qty", "price", "status"]
                )
                for row in reader:
                    if row["action"] == "BUY" and row["status"] in ("EXECUTED", "ORDER_PLACED"):
                        try:
                            ts = parser.parse(row["timestamp"]).date()
                            # always record the most recent buy date
                            if (row["symbol"] not in purchase_dates) or (ts > purchase_dates[row["symbol"]]):
                                purchase_dates[row["symbol"]] = ts
                        except Exception:
                            continue
        except Exception as e:
            self.logger.error(f"Error reading trade log for holding ages: {e}")

        return purchase_dates
