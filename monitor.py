import time
import pytz
import gspread
import pandas as pd
import requests
from datetime import datetime
from kiteconnect import KiteConnect
from utils import load_credentials, send_telegram, get_cnc_holdings, analyze_exit_signals, get_live_price
from indicators import calculate_trailing_sl, check_supertrend_flip
from sheets import log_exit_to_sheet
from holdings_state import load_previous_exits, update_exit_log

# Initialize
secrets = load_credentials()
creds = secrets["zerodha"]
kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(creds["access_token"])
IST = pytz.timezone('Asia/Kolkata')

# Monitor config
SHEET_NAME = secrets["google"]["sheet_name"]
TELEGRAM_CHAT_ID = secrets["telegram"]["chat_id"]
SPREADSHEET_KEY = secrets["sheets"]["SPREADSHEET_KEY"]
DAILY_MONITOR_TAB = "MonitoredStocks"
EXIT_LOG_FILE = "/root/falah-ai-bot/exited_stocks.json"

# Monitoring loop
def monitor_positions():
    now = datetime.now(IST)
    if now.weekday() >= 5 or now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour >= 15 and now.minute >= 30:
        print(f"⏸ Market closed: {now.strftime('%H:%M')}. Retrying later.")
        return

    print(f"📡 Monitoring started at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Get CNC holdings
    holdings = get_cnc_holdings(kite)
    if not holdings:
        print("❌ No CNC holdings found.")
        return

    # Step 2: Load exited list
    exited = load_previous_exits(EXIT_LOG_FILE)

    # Step 3: Log all holdings with date to sheet
    today_str = now.strftime("%Y-%m-%d")
    gc = gspread.service_account(filename="/root/falah-credentials.json")
    sheet = gc.open_by_key(SPREADSHEET_KEY)
    monitor_tab = sheet.worksheet(DAILY_MONITOR_TAB)

    for stock in holdings:
        symbol = stock["tradingsymbol"]
        quantity = stock["quantity"]
        avg_price = stock["average_price"]

        # Write holding info to sheet even if not exiting
        row = [today_str, symbol, quantity, avg_price, "--", "HOLD"]
        monitor_tab.append_row(row)

        if symbol in exited:
            print(f"🔁 {symbol} already exited. Skipping.")
            continue

        # Step 4: Get current price
        try:
            cmp = get_live_price(kite, symbol)
        except:
            print(f"⚠️ Failed to get CMP for {symbol}")
            continue

        # Step 5: Analyze exit logic
        sl_hit = calculate_trailing_sl(symbol, avg_price, cmp)
        st_flip = check_supertrend_flip(symbol)
        ai_exit = analyze_exit_signals(symbol, avg_price, cmp)

        decision = None
        reason = []

        if sl_hit:
            decision = "EXIT"
            reason.append("Trailing SL hit")
        if st_flip:
            decision = "EXIT"
            reason.append("Supertrend flipped")
        if ai_exit:
            decision = "EXIT"
            reason.append("AI exit signal")

        if decision == "EXIT":
            print(f"🚨 Exiting {symbol} @ {cmp} due to: {', '.join(reason)}")
            update_exit_log(EXIT_LOG_FILE, symbol)
            send_telegram(f"🚨 Auto Exit: {symbol} at ₹{cmp}\nReason: {', '.join(reason)}")
            log_exit_to_sheet(SHEET_NAME, DAILY_MONITOR_TAB, symbol, cmp, reason)

    print("✅ Monitoring complete.")

# Run every 15 minutes
if __name__ == "__main__":
    while True:
        try:
            monitor_positions()
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            send_telegram(f"❌ Monitor crashed: {e}")
        time.sleep(900)  # 15 minutes
