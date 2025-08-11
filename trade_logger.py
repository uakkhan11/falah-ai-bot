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
        
    def log_trade(self, symbol, action, quantity, price, status="EXECUTED", timestamp=None):
        Log a trade event locally (CSV) and optionally to Google Sheets.
    
        Args:
            symbol (str): Trading symbol (e.g., 'RELIANCE').
            action (str): 'BUY' or 'SELL'.
            quantity (int): Number of shares traded.
            price (float): Price at which trade executed.
            status (str): Trade status or notes (e.g., 'ORDER_PLACED', 'EXECUTED', 'ORDER_FAILED').
            timestamp (str, optional): ISO formatted string of trade timestamp.
                                        If None, current time will be used.
        """
        # Set timestamp automatically if not passed
        if not timestamp:
            timestamp = datetime.now().isoformat(timespec="seconds")
    
        # Write to CSV (thread‑safe)
        with self.lock:
            try:
                with open(self.csv_path, mode="a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, symbol, action, quantity, price, status])
            except Exception as e:
                self.logger.error(f"Failed to write trade log CSV: {e}")
    
        # Optionally write to Google Sheets
        if self.gsheet_manager:
            try:
                self.gsheet_manager.append_row(
                    self.gsheet_sheet_name,
                    [timestamp, symbol, action, quantity, price, status]
                )
            except Exception as e:
                self.logger.error(f"Failed to write trade log to Google Sheets: {e}")


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
