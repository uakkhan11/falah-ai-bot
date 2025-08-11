# trade_logger.py

import csv
import os
import threading
from datetime import datetime
import logging

class TradeLogger:
    def __init__(self, csv_path="trade_log.csv", gsheet_manager=None, gsheet_sheet_name="TradeLog"):
        self.csv_path = csv_path
        self.gsheet_manager = gsheet_manager
        self.gsheet_sheet_name = gsheet_sheet_name
        self.lock = threading.Lock()  # To write safely in multi-threaded context

        # Setup file if not exists
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "symbol", "action", "quantity", "price", "status"])

        # Configure logging
        self.logger = logging.getLogger(__name__)
        
    def log_trade(self, symbol, action, quantity, price, status="EXECUTED"):
        """
        Log a trade event locally and optionally to Google Sheets.

        Args:
            symbol (str): Trading symbol.
            action (str): "BUY" or "SELL".
            quantity (int): Number of shares.
            price (float): Price at which trade executed.
            status (str): Trade status or notes.
        """

        timestamp = datetime.now().isoformat(timespec="seconds")

        # Write to CSV locally in thread-safe way
        with self.lock:
            try:
                with open(self.csv_path, mode="a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, symbol, action, quantity, price, status])
            except Exception as e:
                self.logger.error(f"Failed to write trade log CSV: {e}")

        # Also optionally log to Google Sheet if available
        if self.gsheet_manager:
            try:
                # Append one row to the configured Google Sheet tab
                self.gsheet_manager.append_row(
                    self.gsheet_sheet_name,
                    [timestamp, symbol, action, quantity, price, status]
                )
            except Exception as e:
                self.logger.error(f"Failed to write trade log to Google Sheets: {e}")
