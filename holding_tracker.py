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

    # holding_tracker.py (add inside the HoldingTracker class)

import json

def _state_path(self):
    # Reuse an adjacent JSON next to the trade log, or set explicitly in ctor
    # If your class already stores a path, replace this function and references accordingly.
    base = os.path.dirname(self.tradelogcsv) if hasattr(self, "tradelogcsv") else "."
    return os.path.join(base, "positions.json")

def load(self):
    """
    Return current positions dict:
      { "SYMBOL": {"qty": int, "entry_price": float or None, "entry_date": "YYYY-MM-DD", "gtt_id": optional} }
    Creates empty file if missing.
    """
    path = self._state_path()
    if not os.path.exists(path):
        try:
            with open(path, "w") as f:
                json.dump({}, f)
        except Exception:
            pass
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data or {}
    except Exception as ex:
        logging.error(f"HoldingTracker.load failed: {ex}")
        return {}

def save(self, positions=None):
    """
    Persist positions dict to JSON. If positions is None, persists internal cache if present,
    else reads, merges, and writes current on-disk content unchanged.
    """
    path = self._state_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        to_write = positions if positions is not None else getattr(self, "_cache_positions", None)
        if to_write is None:
            # noop if nothing to write
            return
        with open(path, "w") as f:
            json.dump(to_write, f, indent=2)
    except Exception as ex:
        logging.error(f"HoldingTracker.save failed: {ex}")

def add(self, symbol, qty, entry_price=None, entry_date=None, gtt_id=None):
    """
    Add/update a long position entry locally.
    """
    pos = self.load()
    pos[symbol] = {
        "qty": int(qty),
        "entry_price": float(entry_price) if entry_price is not None else None,
        "entry_date": entry_date or datetime.now().date().isoformat(),
        "gtt_id": gtt_id
    }
    self._cache_positions = pos
    self.save(pos)

def remove(self, symbol):
    """
    Remove a symbol from local positions.
    """
    pos = self.load()
    if symbol in pos:
        pos.pop(symbol, None)
        self._cache_positions = pos
        self.save(pos)

def get_qty(self, symbol):
    """
    Return current quantity for symbol or 0.
    """
    pos = self.load()
    try:
        return int(pos.get(symbol, {}).get("qty", 0))
    except Exception:
        return 0

def update_gtt(self, symbol, gtt_id):
    """
    Persist GTT id for a symbol.
    """
    pos = self.load()
    if symbol in pos:
        pos[symbol]["gtt_id"] = gtt_id
        self._cache_positions = pos
        self.save(pos)

def get_gtt(self, symbol):
    """
    Return stored GTT id or None.
    """
    pos = self.load()
    return pos.get(symbol, {}).get("gtt_id")

