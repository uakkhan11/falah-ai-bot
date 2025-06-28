import time
import pytz
import gspread
import traceback
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
    now = datetime.now(IST)
    market_open = now.weekday() < 5 and (
        (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
        and (now.hour < 15 or (now.hour == 15 and now.minute < 30))
    )
    today_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"📡 [{timestamp}] Monitoring started | Market open: {market_open}")

    holdings = get_cnc_holdings(kite)
    if not holdings:
        print("❌ No CNC holdings found.")
        return
    print(f"✅ CNC holdings received: {len(holdings)}")

    exited = load_previous_exits(EXIT_LOG_FILE)

    try:
        gc = gspread.service_account(filename="/root/falah-credentials.json")
        sheet = gc.open_by_key(SPREADSHEET_KEY)
        monitor_tab = sheet.worksheet(DAILY_MONITOR_TAB)
        print("✅ Connected to Google Sheet.")
    except Exception as e:
        print(f"❌ Google Sheets error: {e}")
        send_telegram(f"❌ Monitor error connecting to Sheets:\n{e}")
        return

    existing_rows = monitor_tab.get_all_records()
    already_logged = {
        (row.get("Date"), row.get("Symbol"))
        for row in existing_rows
        if row.get("Date") and row.get("Symbol")
    }

    for stock in holdings:
        symbol = stock.get("tradingsymbol") or stock.get("symbol")
        quantity = stock.get("quantity")
        avg_price = stock.get("average_price")

        if not market_open:
            print(f"⏸️ Market closed. Skipping exit checks for {symbol}.")
            continue

        try:
            cmp = get_live_price(kite, symbol)
            if not cmp:
                raise ValueError("CMP unavailable")
        except Exception as e:
            print(f"⚠️ Could not get CMP for {symbol}: {e}")
            continue

        exposure = round(cmp * quantity, 2)

        if (today_str, symbol) not in already_logged:
            row = [today_str, symbol, quantity, avg_price, cmp, exposure, "HOLD"]
            try:
                monitor_tab.append_row(row)
                print(f"📝 Logged {symbol}: Qty={quantity}, Avg={avg_price}, CMP={cmp}, Exposure={exposure}")
            except Exception as e:
                print(f"❌ Failed to log {symbol}: {e}")

        # 🟢 THIS CHECK MUST BE INSIDE THE LOOP:
        last_exit_date = exited.get(symbol)
        if last_exit_date == today_str:
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
            reasons.append(f"ATR SL hit (SL: ₹{sl_price})")
        if st_flip_daily and st_flip_15m:
            reasons.append("Supertrend flipped (daily+15m)")
        if rsi_div:
            reasons.append("RSI bearish divergence")
        if vwap_cross:
            reasons.append("VWAP cross-down")
        if ai_exit:
            reasons.append("AI exit signal")

        if reasons:
            reason_str = ",_
