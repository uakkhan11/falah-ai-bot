import time
import pytz
import gspread
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
SPREADSHEET_KEY = secrets["sheets"]["SPREADSHEET_KEY"]
TELEGRAM_CHAT_ID = secrets["telegram"]["chat_id"]
DAILY_MONITOR_TAB = "MonitoredStocks"
EXIT_LOG_FILE = "/root/falah-ai-bot/exited_stocks.json"

def monitor_positions():
    now = datetime.now(IST)
    market_open = now.weekday() < 5 and (
        (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
        and (now.hour < 15 or (now.hour == 15 and now.minute < 30))
    )
    today_str = now.strftime("%Y-%m-%d")
    print(f"📡 Monitoring started at {now.strftime('%Y-%m-%d %H:%M:%S')} | Market open: {market_open}")

    # Step 1: CNC Holdings
    holdings = get_cnc_holdings(kite)
    print("🔍 Holdings raw output:", holdings)
    if not holdings:
        print("❌ No CNC holdings found.")
        return
    print(f"✅ CNC holdings received: {len(holdings)} stocks")

    # Step 2: Load exited list
    exited = load_previous_exits(EXIT_LOG_FILE)

    # Step 3: Google Sheet setup
    gc = gspread.service_account(filename="/root/falah-credentials.json")
    sheet = gc.open_by_key(SPREADSHEET_KEY)
    monitor_tab = sheet.worksheet(DAILY_MONITOR_TAB)
    print("✅ Google Sheet connected. Ready to log.")

    # Step 4: Fetch existing rows for deduplication
    existing_rows = monitor_tab.get_all_records()
    already_logged = set(
        (row.get("Date"), row.get("Symbol"))
        for row in existing_rows
        if row.get("Date") and row.get("Symbol")
    )

    # Step 5: Loop through each holding
    for stock in holdings:
        symbol = stock.get("tradingsymbol") or stock.get("symbol")
        quantity = stock.get("quantity")
        avg_price = stock.get("average_price")

        if (today_str, symbol) in already_logged:
            print(f"🔁 Already logged today: {symbol}. Skipping.")
            continue

        # Log to Google Sheet
        row = [today_str, symbol, quantity, avg_price, "--", "HOLD"]
        try:
            monitor_tab.append_row(row)
            print(f"📝 Logged {symbol}: Qty={quantity}, Avg={avg_price}")
        except Exception as e:
            print(f"❌ Failed to log {symbol}: {e}")

        if symbol in exited:
            print(f"🔁 {symbol} already exited. Skipping.")
            continue

        # Step 6: Analyze exit logic only during market hours
        if market_open:
            try:
                cmp = get_live_price(kite, symbol)
            except Exception:
                print(f"⚠️ Failed to get CMP for {symbol}")
                continue

            # Trailing SL logic
            sl_price = calculate_trailing_sl([avg_price, cmp])
            sl_hit = sl_price and cmp <= sl_price

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
                reason_str = ", ".join(reason)  # convert list to string
                print(f"🚨 Exiting {symbol} @ {cmp} due to: {reason_str}")
                update_exit_log(EXIT_LOG_FILE, symbol)
                send_telegram(f"🚨 Auto Exit: {symbol} at ₹{cmp}\nReason: {reason_str}")
                log_exit_to_sheet(SHEET_NAME, DAILY_MONITOR_TAB, symbol, cmp, reason_str)

    print("✅ Monitoring complete.")

# Main loop
if __name__ == "__main__":
    while True:
        try:
            monitor_positions()
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            send_telegram(f"❌ Monitor crashed: {e}")
        time.sleep(900)  # Run every 15 min
