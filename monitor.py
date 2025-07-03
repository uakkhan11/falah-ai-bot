# monitor.py

import time
import json
import pytz
import gspread
from datetime import datetime
from kiteconnect import KiteConnect
from app import API_KEY

from ws_live_prices import start_websockets
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

def monitor_positions(kite):
    now = datetime.now(IST)
    market_open = now.weekday() < 5 and (
        (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
        and (now.hour < 15 or (now.hour == 15 and now.minute < 30))
    )
    today_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Market open: {market_open}")
    
        holdings = get_cnc_holdings(kite)
    except Exception as e:
        print(f"❌ Error fetching CNC holdings: {e}")
        holdings = []

    if not holdings:
        print("❌ No CNC holdings found.")
        return
    print(f"✅ CNC holdings received: {len(holdings)}")

    exited = load_previous_exits(EXIT_LOG_FILE)
    if not isinstance(exited, dict):
        print("⚠️ exited_stocks.json invalid format, resetting.")
        exited = {}

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

        token = token_map.get(symbol)
        if token is None:
            print(f"⚠️ No token for {symbol}. Skipping.")
            continue

        cmp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]
        if not cmp:
            print(f"⚠️ No live LTP for {symbol}. Skipping.")
            continue
        print(f"✅ Live CMP for {symbol}: {cmp}")

        # Skip if already exited today
        if exited.get(symbol) == today_str:
            print(f"🔁 {symbol} already exited today. Skipping.")
            continue

        # Check exit signals
        sl_price = calculate_atr_trailing_sl(kite, symbol, cmp)
        sl_hit = sl_price and cmp <= sl_price
        st_flip_daily = check_supertrend_flip(kite, symbol, timeframe="day")
        st_flip_15m = check_supertrend_flip(kite, symbol, timeframe="15minute")
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
                print("✅ Logged exit for TradeLog to sheet.")
            except Exception as e:
                print(f"⚠️ Failed to log exit to sheet: {e}")
        else:
            print(f"✅ {symbol}: No exit criteria met. Holding position.")

    print("✅ Monitoring complete.\n")

if __name__ == "__main__":
    # Load access token fresh
    with open("/root/falah-ai-bot/access_token.json", "r") as f:
        access_token = json.load(f)["access_token"]
    print("✅ Access token loaded.")

    kite = KiteConnect(api_key=creds.get("api_key") or creds.get("api"))
    kite.set_access_token(access_token)
        try:
        profile = kite.profile()
        print("✅ KiteConnect Profile:", profile)
    except Exception as e:
        print(f"❌ API Key/Access Token invalid: {e}")
        return


    # Start WebSockets
    token_list = [int(t) for t in token_map.values()]
    start_websockets(API_KEY, access_token, token_list)

    # Loop monitoring
    while True:
        monitor_positions(kite)
        time.sleep(900)
