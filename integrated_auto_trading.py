import os
import json
import time
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
import gspread

from stock_analysis import analyze_stock
from my_confidence_functions import (
    compute_confidence_score,
    compute_allocation_weight,
    compute_quantity,
    compute_trailing_stop,
    check_exposure_limits
)

# 🟢 Load credentials
with open("/root/falah-ai-bot/secrets.json", "r") as f:
    secrets = json.load(f)

API_KEY = secrets["zerodha"]["api_key"]
ACCESS_TOKEN = secrets["zerodha"]["access_token"]
SPREADSHEET_KEY = secrets["google"]["spreadsheet_key"]

# 🟢 Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# 🟢 Authenticate Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("falah-credentials.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SPREADSHEET_KEY)
log_sheet = sheet.worksheet("TradeLog")

# 🟢 SETTINGS
TOTAL_CAPITAL = 200000
MAX_TRADES = 5
MAX_TOTAL_EXPOSURE = 200000
MAX_SECTOR_EXPOSURE = 100000

# You can define this mapping yourself
symbol_sector_map = {
    "INFY": "IT",
    "TCS": "IT",
    "HDFCBANK": "Banking",
    # ...
}

# 🟢 Main Loop
def auto_trade(symbols):
    current_exposure = 0
    sector_exposure = {}

    for symbol in symbols:
        print(f"\n🔍 Analyzing {symbol}")
        try:
            result = analyze_stock(kite, symbol)
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            continue

        # Compute metrics
        confidence = compute_confidence_score(result)
        atr_percent = (result["atr"] / result["cmp"]) * 100
        allocation_weight = compute_allocation_weight(confidence, atr_percent)
        qty = compute_quantity(TOTAL_CAPITAL, MAX_TRADES, allocation_weight, result["cmp"])

        if qty <= 0:
            print(f"⚠️ Skipping {symbol} due to low confidence or sizing.")
            continue

        trade_value = qty * result["cmp"]
        sector = symbol_sector_map.get(symbol, "Unknown")

        if not check_exposure_limits(current_exposure, trade_value, sector_exposure, sector, MAX_TOTAL_EXPOSURE, MAX_SECTOR_EXPOSURE):
            print(f"⚠️ Exposure limit exceeded for {symbol}. Skipping.")
            continue

        # Place order
        try:
            kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=qty,
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_CNC
            )
            print(f"✅ Order placed: {symbol} Qty: {qty}")

            # Update exposure
            current_exposure += trade_value
            sector_exposure[sector] = sector_exposure.get(sector, 0) + trade_value

            # Compute initial trailing stop
            trailing_sl = compute_trailing_stop(result["cmp"], result["atr"], result["cmp"])

            # Log to Google Sheets
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_sheet.append_row([
                timestamp, symbol, "BUY", qty, result["cmp"], trailing_sl,
                f"{confidence:.2f}", f"{allocation_weight:.2f}", "Auto-Traded"
            ])
            print(f"📝 Logged to Google Sheets.")
        except Exception as e:
            print(f"❌ Error placing order for {symbol}: {e}")

if __name__ == "__main__":
    # Example symbols (you can load dynamically)
    candidate_symbols = ["INFY", "TCS", "HDFCBANK", "ICICIBANK", "SBIN"]
    auto_trade(candidate_symbols)
