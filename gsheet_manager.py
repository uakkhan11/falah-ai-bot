# gsheet_manager.py

import gspread
import logging
import os
import json
import datetime
from typing import Dict
from google.oauth2.service_account import Credentials

try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

class GSheetManager:
    def __init__(self, cfg):
        self.enabled = gspread is not None and Credentials is not None
        self.cfg = cfg
        if not self.enabled:
            return
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(cfg.gs_service_account_json, scopes=scopes)
        self.gc = gspread.authorize(creds)
        self.sheet = self.gc.open_by_key(cfg.gs_spreadsheet_id)
        self.ws = self.sheet.worksheet(cfg.gs_worksheet_name)

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

    def append_row(self, row: Dict):
        """
        Append a dict as a row; keys must match the header row.
        If header mismatch, falls back to ordered values by sorted keys.
        """
        if not self.enabled:
            return False
        try:
            headers = self.ws.row_values(1)
            if headers:
                values = [str(row.get(h, "")) for h in headers]
            else:
                # No header; append sorted key order
                keys_sorted = sorted(row.keys())
                values = [str(row[k]) for k in keys_sorted]
            self.ws.append_row(values)
            return True
        except Exception as ex:
            # Soft-fail to keep orchestration running
            return False
