import time
import pytz
import gspread
from datetime import datetime
from kiteconnect import KiteConnect
from utils import (
    load_credentials,
    send_telegram,
    get_cnc_holdings,
    analyze_exit_signals,
    get_live_price,
)
from indicators import (
    calculate_atr_trailing_sl,
    check_supertrend_flip,
    check_rsi_bearish_divergence,
    check_vwap_cross,
)
from sheets import log_exit_to_sheet
from holdings_state import load_previous_exits, update_exit_log

# Initialize
print("✅ Starting monitor.py...")
secrets = load_credentials()
creds = secrets["zerodha"]
kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(creds["access_token"])
IST = pytz.timezone("Asia/Kolkata")

SHEET_NAME = secrets["google"]["sheet_name"]
SPREADSHEET_KEY = secrets["sheets"]["SPREADSHEET_KEY"]
DAILY_MONITOR_TAB = "MonitoredStocks"
EXIT_LOG_FILE = "/root/falah-ai-bot/exited_stocks.json"

def monitor_positions():
    print("🚀 monitor_positions() started")

    now = datetime.now(IST)
    market_open = now.weekday() < 5 and (
        (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
        and (now.hour < 15 or (now.hour == 15 and now.minute < 30))
    )
    today_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"📡 [{timestamp}] Market open: {market_open}")

    holdings = get_cnc_holdings(kite)
    print("✅ get_cnc_holdings() returned:", holdings)

    if not holdings:
        print("❌ No CNC holdings found.")
        return
    print(f"✅ CNC holdings received: {len(holdings)}")

    exited = load_previous_exits(EXIT_LOG_FILE)
    print("✅ Loaded exited stocks log.")

    gc = gspread.service_account(filename="/root/falah-credentials.json")
    sheet = gc.open_by_key(SPREADSHEET_KEY)
    monitor_tab = sheet.worksheet(DAILY_MONITOR_TAB)
    print("✅ Connected to Google Sheet.")

    existing_rows = monitor_tab.get_all_records()
    print(f"✅ Loaded {len(existing_rows)} existing rows from sheet.")

    for stock in holdings:
        symbol = stock.get("tradingsymbol") or stock.get("symbol")
        quantity = stock.get("quantity")
        avg_price = stock.get("average_price")

        print(f"\n🔍 Processing {symbol} (Qty={quantity}, Avg={avg_price})")

        # Always get CMP for logging
        try:
            cmp = get_live_price(kite, symbol)
            if not cmp:
                raise ValueError("CMP unavailable")
            print(f"✅ Live CMP for {symbol}: {cmp}")
        except Exception as e:
            print(f"⚠️ Could not get CMP for {symbol}: {e}")
            cmp = "--"

        exposure = round(cmp * quantity, 2) if cmp != "--" else "--"

        # Search for existing row
        row_idx = None
        for idx, row in enumerate(existing_rows, start=2):
            if row.get("Date") == today_str and row.get("Symbol") == symbol:
                row_idx = idx
                break

        if row_idx:
            # Update CMP and Exposure
            try:
                monitor_tab.update(f"E{row_idx}", [[cmp]])
                monitor_tab.update(f"F{row_idx}", [[exposure]])
                print(f"🔄 Updated CMP/Exposure in sheet: CMP={cmp}, Exposure={exposure}")
            except Exception as e:
                print(f"⚠️ Failed to update CMP/Exposure: {e}")
        else:
            # Append new row
            row = [today_str, symbol, quantity, avg_price, cmp, exposure, "HOLD"]
            try:
                monitor_tab.append_row(row)
                print(f"📝 Added new row for {symbol}.")
            except Exception as e:
                print(f"❌ Failed to log {symbol}: {e}")

        if not market_open:
            print(f"⏸️ Market closed. Skipping exit checks for {symbol}.")
            continue

        last_exit_date = exited.get(symbol)
        if last_exit_date == today_str:
            print(f"🔁 {symbol} already exited today. Skipping.")
            continue

        sl_price = calculate_atr_trailing_sl(kite, symbol, cmp)
        sl_hit = sl_price and cmp <= sl_price

        st_flip_daily = check_supertrend_flip(symbol, interval="day")
        st_flip_15m = check_supertrend_flip(symbol
