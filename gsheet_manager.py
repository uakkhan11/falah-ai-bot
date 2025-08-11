# gsheet_manager.py
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
import logging

class GSheetManager:
    def __init__(self, credentials_file="falah-credentials.json"):
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_file(credentials_file, scopes=scope)
        self.gc = gspread.authorize(creds)
        self.logger = logging.getLogger(__name__)

    def get_symbols_from_sheet(self, sheet_key_or_url, worksheet_name="HalalList"):
        try:
            if sheet_key_or_url.startswith("http"):
                sh = self.gc.open_by_url(sheet_key_or_url)
            else:
                sh = self.gc.open_by_key(sheet_key_or_url)
            ws = sh.worksheet(worksheet_name)
            values = ws.col_values(1)[1:]  # skip header
            symbols = [v.strip().upper() for v in values if v.strip()]
            self.logger.info(f"Loaded {len(symbols)} symbols from Google Sheet")
            return symbols
        except Exception as e:
            self.logger.error(f"Error reading Google Sheet: {e}")
            return []
