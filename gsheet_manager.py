# gsheet_manager.py

import gspread
import logging
from google.oauth2.service_account import Credentials

class GSheetManager:
    def __init__(self, credentials_file="falah-credentials.json", sheet_key=None):
        # Define the scopes
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        # Authorize with service account credentials
        creds = Credentials.from_service_account_file(credentials_file, scopes=scope)
        self.gc = gspread.authorize(creds)
        self.logger = logging.getLogger(__name__)

        # Store sheet key (can be set dynamically too)
        self.sheet_key = sheet_key

    def set_sheet_key(self, sheet_key):
        """Set or change the sheet key dynamically."""
        self.sheet_key = sheet_key

    def get_symbols_from_sheet(self, sheet_key=None, worksheet_name="HalalList"):
        """
        Fetch the list of trading symbols from the specified Google Sheet worksheet.
        If sheet_key is not passed, uses the one stored during initialization.
        """
        try:
            key_to_use = sheet_key or self.sheet_key
            if not key_to_use:
                raise ValueError("No Google Sheet key provided or set in GSheetManager.")

            sh = self.gc.open_by_key(key_to_use)
            ws = sh.worksheet(worksheet_name)
            values = ws.col_values(1)[1:]  # skip header
            symbols = [v.strip().upper() for v in values if v.strip()]
            self.logger.info(f"Loaded {len(symbols)} symbols from Google Sheet")
            return symbols
        except Exception as e:
            self.logger.error(f"Error reading Google Sheet: {e}")
            return []

    def append_row(self, worksheet_name, row):
        """
        Append a single row to a worksheet.
        Requires sheet_key to be set or passed during initialization.
        """
        try:
            if not self.sheet_key:
                raise ValueError("No Google Sheet key is set for append_row()")
            sh = self.gc.open_by_key(self.sheet_key)
            ws = sh.worksheet(worksheet_name)
            ws.append_row(row)
            self.logger.info(f"Appended row to {worksheet_name}: {row}")
            return True
        except Exception as e:
            self.logger.error(f"Append row failed on sheet {worksheet_name}: {e}")
            return False
