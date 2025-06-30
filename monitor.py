import time
import json
import pytz
import gspread
from datetime import datetime
from kiteconnect import KiteConnect

from ws_live_prices import live_prices, start_websocket
from utils import (
    load_credentials,
    send_telegram,
    get_cnc_holdings,
    analyze_exit_signals,
)
from indicators import (
    calculate_atr_trailing_sl,
    check_supertrend_flip,
    check_rsi_bearish_divergence,
    check_vwap_cross,
)
from sheets import log_exit_to_sheet
from holdings_state import load_previous_exits, update_exit_log

print("✅ All imports finished")

# Load credentials
secrets = load_credentials()
creds = secrets["zerodha"]
print("✅ Credentials loaded")

# Initialize Kite
kite = KiteConnect(api_key=creds["api_key"])
kite.set_access_token(creds["access_token"])
print("✅ KiteConnect initialized")

IST = pytz.timezone("Asia/Kolkata")

SHEET_NAME = secrets["google"]["sheet_name"]
SPREADSHEET_KEY = secrets["sheets"]["SPREADSHEET_KEY"]
DAILY_MONITOR_TAB = "MonitoredStocks"
EXIT_LOG_FILE = "/root/falah-ai-bot/exited_stocks.json"

# Load instrument tokens
try:
    with open("/root/falah-ai-bot/tokens.json", "r") as f:
        token_map = json.load(f)
    print(f"✅ Loaded {len(token_map)} instrument tokens.")
except Exception as e:
    print(f"⚠️ Could not load tokens.json: {e}")
    token_map = {}

def monitor_positions():
    now = datetime.now(IST)
    market_open = now.weekday() < 5 and (
        (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
        and (now.hour < 15 or (now.hour == 15 and now.minute < 30))
    )
    today_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Market open: {market_open}")

    holdings = get_cnc_holdings(kite)
    print("Holdings returned:", holdings)

    if not holdings:
        print("❌ No CNC holdings found.")
        return
    print(f"✅ CNC holdings received: {len(holdings)}")

    exited = load_previous_exits(EXIT_LOG_FILE)
    print(f"✅ Loaded exited stocks log ({len(exited)} exited symbols).")

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

        print(f"🔍 Processing {symbol} (Qty={quantity}, Avg={avg_price})")

        # Use WebSocket LTP
        token = token_map.get(symbol)
        if token is None:
            print(f"⚠️ No token for {symbol}. Skipping.")
            continue

        cmp = live_prices.get(int(token))
        if not cmp:
            print(f"⚠️ No live LTP for {symbol}. Skipping.")
            continue
        print(f"✅ Live CMP for {symbol}: {cmp}")

        exposure = round(cmp * quantity, 2)

        # Check if already logged
        row_idx = None
        for idx, row in enumerate(existing_rows, start=2):
            if row.get("Date") == today_str and row.get("Symbol") == symbol:
                row_idx = idx
                break

        if row_idx:
            try:
                monitor_tab.update("E"+str(row_idx), [[cmp]])
                monitor_tab.update("F"+str(row_idx), [[exposure]])
                print(f"🔄 Updated CMP/Exposure: CMP={cmp}, Exposure={exposure}")
            except Exception as e:
                print(f"⚠️ Failed to update CMP/Exposure: {e}")
        else:
            # Double-check before append
            latest_rows = monitor_tab.get_all_records()
            duplicate = any(
                r.get("Date") == today_str and r.get("Symbol") == symbol
                for r in latest_rows
            )
            if duplicate:
                print(f"⚠️ Detected duplicate row for {symbol}. Skipping append.")
                continue

            row = [today_str, symbol, quantity, avg_price, cmp, exposure, "HOLD"]
            try:
                monitor_tab.append_row(row)
                print(f"📝 Added new row for {symbol}.")
            except Exception as e:
                print(f"❌ Failed to log {symbol}: {e}")

        if not market_open:
            print(f"⏸️ Market closed. Skipping exit checks for {symbol}.")
            continue

        # Corrected exit check (use `in`)
        if symbol in exited:
            print(f"🔁 {symbol} already exited today. Skipping.")
            continue

        sl_price = calculate_atr_trailing_sl(kite, symbol, cmp)
        sl_hit = sl_price and cmp <= sl_price

        st_flip_daily = check_supertrend_flip(symbol, interval="day")
        st_flip_15m = check_supertrend_flip(symbol, interval="15minute")

        rsi_div = check_rsi_bearish_divergence(kite, symbol)
        vwap_cross = check_vwap_cross(kite, symbol)

        ai_exit = analyze_exit_signals(symbol, avg_price, cmp)

        reasons = []
        if sl_hit:
            reasons.append(f"ATR SL hit (SL: {sl_price})")
        if st_flip_daily and st_flip_15m:
            reasons.append("Supertrend flipped (daily+15m)")
        if rsi_div:
            reasons.append("RSI bearish divergence")
        if vwap_cross:
            reasons.append("VWAP cross-down")
        if ai_exit:
            reasons.append("AI exit signal")

        if reasons:
            reason_str = ", ".join(reasons)
            print(f"🚨 Exit triggered for {symbol} at {cmp}: {reason_str}")
            update_exit_log(EXIT_LOG_FILE, symbol)
            send_telegram(
                f"🚨 Auto Exit Triggered\n"
                f"Symbol: {symbol}\n"
                f"Price: {cmp}\n"
                f"Reasons: {reason_str}"
            )
            try:
                log_exit_to_sheet(SHEET_NAME, DAILY_MONITOR_TAB, symbol, cmp, reason_str)
            except Exception as e:
                print(f"⚠️ Failed to log exit to sheet: {e}")
        else:
            print(f"✅ {symbol}: No exit criteria met. Holding position.")

    print("✅ Monitoring complete.\n")

if __name__ == "__main__":
    # Start WebSocket streaming
    token_list = [int(token) for token in token_map.values()]
    start_websocket(token_list)

    # Loop monitoring every 15 min
    while True:
        monitor_positions()
        time.sleep(900)
