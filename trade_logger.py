# trade_logger.py

import csv
import os
from datetime import datetime

class TradeLogger:
    """
    Logs trade events to a local CSV file and optionally to Google Sheets via GSheetManager.
    """

    def __init__(self, csv_path, gsheet_manager=None, gsheet_sheet_name=None):
        """
        Args:
            csv_path (str): Path to the local CSV file for logging trades.
            gsheet_manager (GSheetManager, optional): Manager to log trades to Google Sheets.
            gsheet_sheet_name (str, optional): Name of the sheet in the Google Sheet document.
        """
        self.csv_path = csv_path
        self.gsheet = gsheet_manager
        self.sheet_name = gsheet_sheet_name

        # Ensure CSV file exists with headers
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price',
                    'status', 'notes'
                ])

    def log_trade(self, symbol, side, quantity, price, status, notes=''):
        """
        Log a trade event locally (CSV) and optionally to Google Sheets.

        Args:
            symbol (str): Trading symbol, e.g., 'RELIANCE'
            side (str): 'BUY' or 'SELL'
            quantity (int): Number of shares/contracts
            price (float): Execution price
            status (str): e.g., 'ORDER_PLACED', 'ORDER_FAILED', 'EXIT_Profit_Target'
            notes (str, optional): Any additional notes or reason
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Append to local CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, symbol, side, quantity, price, status, notes])

        # Optionally append to Google Sheet
        if self.gsheet and self.sheet_name:
            try:
                self.gsheet.append_row(
                    sheet_name=self.sheet_name,
                    row=[timestamp, symbol, side, quantity, price, status, notes]
                )
            except Exception as e:
                # If Google Sheets logging fails, just print the error
                print(f"⚠️ Failed to log trade to Google Sheets: {e}")

