# app.py
import time
import json
import pytz
from config import load_secrets
from datetime import datetime
from credentials import get_kite, validate_kite, load_secrets
from data_fetch import get_cnc_holdings, get_live_ltp
from ai_engine import analyze_exit_signals
from sheets import log_exit_to_sheet
from utils import send_telegram

IST = pytz.timezone("Asia/Kolkata")

def monitor_once(kite, token_map, sheet_name, spreadsheet_key, exit_log_file):
    now = datetime.now(IST)
    today_str = now.strftime("%Y-%m-%d")

    holdings = get_cnc_holdings(kite)
    if not holdings:
        print("❌ No CNC holdings.")
        return

    for stock in holdings:
        symbol = stock["tradingsymbol"]
        quantity = stock["quantity"]
        avg_price = stock["average_price"]

        print(f"🔍 {symbol}: Qty={quantity} Avg={avg_price}")

        token = token_map.get(symbol)
        if not token:
            print(f"⚠️ No token for {symbol}. Skipping.")
            continue

        cmp = get_live_ltp(kite, symbol)
        print(f"✅ CMP {symbol}: {cmp}")

        reasons = analyze_exit_signals(kite, symbol, avg_price, cmp)

        if reasons:
            reason_str = ", ".join(reasons)
            print(f"🚨 Exit triggered: {reason_str}")
            send_telegram(
                f"🚨 Exit Triggered\nSymbol: {symbol}\nPrice: {cmp}\nReasons: {reason_str}"
            )
            log_exit_to_sheet(sheet_name, "MonitoredStocks", symbol, cmp, reason_str)
        else:
            print(f"✅ No exit for {symbol}.")


if __name__ == "__main__":
    # Load credentials
    secrets = load_secrets()

    print("API Key:", secrets["zerodha"]["api_key"])
    print("Access Token:", secrets["zerodha"]["access_token"])

    kite = get_kite()

    if not validate_kite(kite):
        send_telegram("❌ Falāh Bot: Invalid token, exiting.")
        exit(1)

    # Load tokens
    with open("/root/falah-ai-bot/secrets.json", "r") as f:
        token_map = json.load(f)
        # Zerodha credentials
    API_KEY = secrets["zerodha"]["api_key"]
    API_SECRET = secrets["zerodha"]["api_secret"]
    ACCESS_TOKEN = secrets["zerodha"]["access_token"]

# Google Sheets credentials
SPREADSHEET_KEY = secrets["google"]["spreadsheet_key"]

# If you also have the JSON key file for gspread:
CREDS_JSON = "/root/falah-ai-bot/falah-credentials.json

    exit_log_file = "/root/falah-ai-bot/exited_stocks.json"
    sheet_name = secrets["google"]["sheet_name"]
    spreadsheet_key = secrets["sheets"]["SPREADSHEET_KEY"]

    # Loop monitoring
    while True:
        try:
            monitor_once(kite, token_map, sheet_name, spreadsheet_key, exit_log_file)
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            if "Incorrect `api_key` or `access_token`" in str(e):
                send_telegram("❌ Falāh Bot: Token expired, exiting.")
                exit(1)
        time.sleep(900)
