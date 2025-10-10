# gsheet_manager.py

import os
from typing import Dict

# Optional deps; orchestrator should continue even if not installed
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

class GoogleSheetLogger:
    def __init__(self, cfg):
        self.enabled = bool(gspread and Credentials
                            and getattr(cfg, "gs_service_account_json", None)
                            and getattr(cfg, "gs_spreadsheet_id", None)
                            and getattr(cfg, "gs_worksheet_name", None))
        self.ws = None
        if not self.enabled:
            return
        try:
            scopes = ["https://www.googleapis.com/auth/spreadsheets"]
            creds = Credentials.from_service_account_file(cfg.gs_service_account_json, scopes=scopes)
            client = gspread.authorize(creds)
            sh = client.open_by_key(cfg.gs_spreadsheet_id)
            self.ws = sh.worksheet(cfg.gs_worksheet_name)
        except Exception:
            # Disable gracefully if setup fails
            self.enabled = False
            self.ws = None

    def append_row(self, row: Dict):
        """
        Append a dict as a row; if header exists, map by header names.
        On any error or if disabled, return False to keep pipeline resilient.
        """
        if not self.enabled or not self.ws:
            return False
        try:
            headers = self.ws.row_values(1)
            if headers:
                values = [str(row.get(h, "")) for h in headers]
            else:
                # Fallback: keys in alpha order
                keys = sorted(row.keys())
                values = [str(row[k]) for k in keys]
            self.ws.append_row(values)
            return True
        except Exception:
            return False
