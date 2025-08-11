import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
import logging

class GSheetManager:
    def __init__(self, credentials_file):
        """Initialize Google Sheets connection"""
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        self.credentials = Credentials.from_service_account_file(
            credentials_file, scopes=scope
        )
        self.gc = gspread.authorize(self.credentials)
        self.logger = logging.getLogger(__name__)
        
    def get_symbols_from_sheet(self, sheet_url_or_key, worksheet_name="Sheet1", symbol_column="A"):
        """
        Get trading symbols from Google Sheet
        
        Args:
            sheet_url_or_key: Google Sheet URL or sheet key
            worksheet_name: Name of the worksheet (default: "Sheet1")
            symbol_column: Column containing symbols (default: "A")
        
        Returns:
            List of symbols to trade
        """
        try:
            # Open the spreadsheet
            if sheet_url_or_key.startswith('http'):
                sheet = self.gc.open_by_url(sheet_url_or_key)
            else:
                sheet = self.gc.open_by_key(sheet_url_or_key)
            
            # Get the worksheet
            worksheet = sheet.worksheet(worksheet_name)
            
            # Get all data from the worksheet
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            
            # Extract symbols (assuming first column contains symbols)
            if df.empty:
                # Fallback: get values from specified column
                symbols = worksheet.col_values(1)[1:]  # Skip header
            else:
                # Get symbols from first column or specified column
                symbols = df.iloc[:, 0].dropna().tolist()
            
            # Clean symbols (remove empty strings, spaces)
            symbols = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
            
            self.logger.info(f"Loaded {len(symbols)} symbols from Google Sheet")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error reading Google Sheet: {e}")
            return []
    
    def get_symbol_data_from_sheet(self, sheet_url_or_key, worksheet_name="Sheet1"):
        """
        Get complete symbol data with additional parameters
        
        Returns:
            DataFrame with symbol data (symbol, sector, market_cap, etc.)
        """
        try:
            if sheet_url_or_key.startswith('http'):
                sheet = self.gc.open_by_url(sheet_url_or_key)
            else:
                sheet = self.gc.open_by_key(sheet_url_or_key)
            
            worksheet = sheet.worksheet(worksheet_name)
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            
            self.logger.info(f"Loaded symbol data: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading symbol data: {e}")
            return pd.DataFrame()
    
    def update_trading_results(self, sheet_url_or_key, worksheet_name, results_data):
        """
        Update Google Sheet with trading results
        
        Args:
            results_data: List of dictionaries with trading results
        """
        try:
            if sheet_url_or_key.startswith('http'):
                sheet = self.gc.open_by_url(sheet_url_or_key)
            else:
                sheet = self.gc.open_by_key(sheet_url_or_key)
            
            worksheet = sheet.worksheet(worksheet_name)
            
            # Convert results to DataFrame
            df = pd.DataFrame(results_data)
            
            # Update the sheet
            worksheet.clear()
            worksheet.update([df.columns.values.tolist()] + df.values.tolist())
            
            self.logger.info(f"Updated {len(results_data)} results to Google Sheet")
            
        except Exception as e:
            self.logger.error(f"Error updating Google Sheet: {e}")
