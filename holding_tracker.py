#!/usr/bin/env python3
# holding_tracker.py
#
# Persistent local portfolio tracker with:
# - load/save of positions JSON
# - add/remove/update helpers
# - get_qty, get_gtt, update_gtt
# - T1/T2/Settled age annotation utility (reuses trade log CSV)
#
# Works with live_orchestrator.py expectations.

import os
import csv
import json
import logging
from datetime import datetime, date
from dateutil import parser

class HoldingTracker:
    def __init__(self, trade_log_csv="trade_log.csv", state_path=None):
        """
        trade_log_csv: path to trade log CSV used for age (T1/T2) utility.
        state_path: explicit JSON file path to store positions, e.g., state/positions.json.
        """
        self.trade_log_csv = trade_log_csv
        self.logger = logging.getLogger(__name__)
        self._positions_path = state_path  # optional override

    # -------------------- Persistence --------------------
    def _state_path(self):
        """
        Determine positions.json location.
        Preference order:
          - explicit state_path passed at construction
          - next to the trade_log_csv file
        """
        if self._positions_path:
            return self._positions_path
        base = os.path.dirname(self.trade_log_csv) or "."
        return os.path.join(base, "positions.json")

    def load(self):
        """
        Load current positions from JSON. Structure:
          {
            "SYMBOL": {
              "qty": int,
              "entry_price": float or None,
              "entry_date": "YYYY-MM-DD",
              "gtt_id": str or None
            },
            ...
          }
        Returns {} if file doesn't exist.
        """
        path = self._state_path()
        if not os.path.exists(path):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
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
            self.logger.error(f"HoldingTracker.load failed: {ex}")
            return {}

    def save(self, positions=None):
        """
        Save positions dict to JSON. If positions is None, uses internal cache if present.
        """
        path = self._state_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            to_write = positions if positions is not None else getattr(self, "_cache_positions", None)
            if to_write is None:
                # nothing to write
                return
            with open(path, "w") as f:
                json.dump(to_write, f, indent=2)
        except Exception as ex:
            self.logger.error(f"HoldingTracker.save failed: {ex}")

    # -------------------- CRUD Helpers --------------------
    def add(self, symbol, qty, entry_price=None, entry_date=None, gtt_id=None):
        """
        Add or update a long position record.
        """
        pos = self.load()
        pos[symbol] = {
            "qty": int(qty),
            "entry_price": float(entry_price) if entry_price is not None else None,
            "entry_date": (entry_date or date.today().isoformat()),
            "gtt_id": gtt_id
        }
        self._cache_positions = pos
        self.save(pos)

    def remove(self, symbol):
        """
        Remove a symbol from positions.
        """
        pos = self.load()
        if symbol in pos:
            pos.pop(symbol, None)
            self._cache_positions = pos
            self.save(pos)

    def get_qty(self, symbol):
        """
        Return current quantity of symbol or 0.
        """
        pos = self.load()
        try:
            return int(pos.get(symbol, {}).get("qty", 0))
        except Exception:
            return 0

    def update_gtt(self, symbol, gtt_id):
        """
        Persist/replace GTT id for a symbol.
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

    # -------------------- Age/T1-T2 Utility --------------------
    def get_holdings_with_age(self, positions):
        """
        Augment positions (list of dicts with 'symbol' and 'qty') with holding age labels:
          - "T1 (today's buy)", "T1", "T2", "Settled", or "Unknown"
        Uses trade_log_csv to determine latest BUY date per symbol.
        """
        purchase_dates = self._load_latest_buy_dates()
        today = date.today()
        result = []

        for pos in positions:
            symbol = pos.get("symbol")
            qty = int(pos.get("qty", 0))
            if qty <= 0:
                continue  # skip shorts/closed

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

            pos = dict(pos)
            pos["holding_status"] = status
            pos["holding_days"] = days
            result.append(pos)

        return result

    def _load_latest_buy_dates(self):
        """
        Parse the trade log CSV and return a dict of {symbol: latest BUY date}.
        Expected CSV headers: timestamp, symbol, side, quantity, price, status, notes
        timestamp must be parseable to datetime.
        """
        purchasedates = {}
        path = self.trade_log_csv
        if not os.path.exists(path):
            return purchasedates

        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        side = (row.get("side") or "").upper()
                        if side != "BUY":
                            continue
                        symbol = row.get("symbol")
                        ts = row.get("timestamp") or row.get("date") or row.get("time")
                        if not symbol or not ts:
                            continue
                        dt = parser.parse(ts)
                        d = dt.date()
                        prev = purchasedates.get(symbol)
                        if (prev is None) or (d > prev):
                            purchasedates[symbol] = d
                    except Exception:
                        continue
        except Exception as ex:
            self.logger.error(f"HoldingTracker._load_latest_buy_dates failed: {ex}")

        return purchasedates
