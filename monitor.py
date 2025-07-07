# monitor.py – Falāh Bot Monitoring Module

import os
import time
import json
import datetime
import pandas as pd
import requests
import gspread
from config import load_secrets
from kiteconnect import KiteConnect
from oauth2client.service_account import ServiceAccountCredentials

# 🟢 Load credentials
with open("/root/falah-ai-bot/secrets.json", "r") as f:
    secrets = json.load(f)

API_KEY = secrets["zerodha"]["api_key"]
API_SECRET = secrets["zerodha"]["api_secret"]
ACCESS_TOKEN = secrets["zerodha"]["access_token"]
BOT_TOKEN = secrets["telegram"]["bot_token"]
CHAT_ID = secrets["telegram"]["chat_id"]
SPREADSHEET_KEY = secrets["global"]["google_sheet_key"]

# 🟢 Authenticate Zerodha
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# 🟢 Authenticate Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("falah-credentials.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SPREADSHEET_KEY)
log_sheet = sheet.worksheet("TradeLog")
monitor_sheet = sheet.worksheet("MonitoredStocks")

# 🟢 Helper: Telegram
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram error: {e}")

# 🟢 Helper: Check Market Hours
def is_market_open():
    now = datetime.datetime.now()
    return now.weekday() < 5 and now.hour >= 9 and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))

# 🟢 Helper: Compute Trailing Stoploss
def calculate_trailing_sl(entry, highest, trail_pct=2):
    return highest * (1 - trail_pct / 100)

# 🟢 Helper: AI Exit Decision
def ai_exit_decision(symbol, ltp, entry, trailing_sl, supertrend_signal, volume_spike, reversal_pattern):
    ai_score = 0
    reasons = []
    if ltp < trailing_sl:
        ai_score += 40
        reasons.append("Trailing SL hit")
    if supertrend_signal == "SELL":
        ai_score += 25
        reasons.append("Supertrend SELL")
    if volume_spike:
        ai_score += 15
        reasons.append("Volume Spike")
    if reversal_pattern:
        ai_score += 15
        reasons.append("Reversal Pattern")
    if ltp < entry * 0.95:
        ai_score += 10
        reasons.append("5% Loss Threshold")
    return ai_score, reasons

# 🟢 Helper: Dummy Supertrend/Volume/Reversal (replace with real logic)
def get_supertrend(symbol):
    return "BUY"  # or "SELL"

def check_volume_spike(symbol):
    return False

def detect_reversal(symbol):
    return False

# 🟢 Main Monitoring Loop
def monitor_positions():
    send_telegram("✅ <b>Falāh Monitoring Started</b>\nTracking all CNC holdings...")
    while True:
        if not is_market_open():
            print("⏸ Market closed. Sleeping 10 min...")
            time.sleep(600)
            continue

        positions = kite.positions()["net"]
        holdings = [p for p in positions if p["product"] == "CNC" and p["quantity"] > 0]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_rows = []

        for pos in holdings:
            symbol = pos["tradingsymbol"]
            qty = pos["quantity"]
            avg_price = pos["average_price"]
            ltp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]

            # Compute trailing SL
            highest = max(ltp, avg_price * 1.03)
            trailing_sl = calculate_trailing_sl(avg_price, highest, trail_pct=2)

            # AI-based exit analysis
            supertrend = get_supertrend(symbol)
            volume_spike = check_volume_spike(symbol)
            reversal = detect_reversal(symbol)
            ai_score, reasons = ai_exit_decision(symbol, ltp, avg_price, trailing_sl, supertrend, volume_spike, reversal)

            print(f"🔍 {symbol}: Qty={qty}, LTP={ltp:.2f}, Trailing SL={trailing_sl:.2f}, AI Score={ai_score}")

            # Log all holdings to sheet
            all_rows.append([
                timestamp,
                symbol,
                qty,
                avg_price,
                ltp,
                trailing_sl,
                ai_score,
                ", ".join(reasons) if reasons else "Holding"
            ])

            # Exit if AI score >= 50
            if ai_score >= 50:
                try:
                    kite.place_order(
                        variety=kite.VARIETY_REGULAR,
                        exchange=kite.EXCHANGE_NSE,
                        tradingsymbol=symbol,
                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                        quantity=qty,
                        order_type=kite.ORDER_TYPE_MARKET,
                        product=kite.PRODUCT_CNC
                    )
                    msg = f"⚠️ <b>Exit Triggered</b>\n{symbol}\nQty: {qty}\nLTP: ₹{ltp:.2f}\nReasons: {', '.join(reasons)}"
                    send_telegram(msg)
                    log_sheet.append_row([
                        timestamp, symbol, "SELL", qty, ltp, "Exit Triggered", ", ".join(reasons)
                    ])
                except Exception as e:
                    send_telegram(f"❌ Error exiting {symbol}: {e}")

        if all_rows:
            monitor_sheet.append_rows(all_rows, value_input_option="USER_ENTERED")

        # Wait 15 minutes
        time.sleep(900)

# 🟢 Run Monitoring
if __name__ == "__main__":
    monitor_positions()
