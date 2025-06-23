# monitor.py – Falāh Bot Live Stock Monitor

import time
import pytz
import gspread
import requests
import traceback
import logging
import pandas as pd
from datetime import datetime
from kiteconnect import KiteConnect
from oauth2client.service_account import ServiceAccountCredentials

# CONFIGS
API_KEY       = "ijzeuwuylr3g0kug"
ACCESS_TOKEN  = "HdiGzilCwOT8yZqCreno4mb7FyJwxMOy"
CREDS_FILE    = "falah-credentials.json"
SHEET_KEY     = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
TELEGRAM_TOKEN = "7763450358:AAH32bWYyu_hXR6l-UaVMaarFGZ4YFOv6q8"
TELEGRAM_CHAT_ID = "6784139148"

# SETUP
logging.basicConfig(level=logging.INFO)

def is_market_open():
    now = datetime.now(pytz.timezone("Asia/Kolkata"))
    return now.weekday() < 5 and datetime.strptime("09:15", "%H:%M").time() <= now.time() <= datetime.strptime("15:30", "%H:%M").time()

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(url, data=data)
    except:
        logging.warning("❌ Telegram failed.")

def init_kite():
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite

def init_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_KEY)

def fetch_positions(kite):
    holdings = kite.holdings()
    tracked = [h for h in holdings if h['quantity'] > 0]
    return tracked

def log_to_sheet(sheet, records):
    try:
        ws = sheet.worksheet("MonitoredStocks")
    except:
        ws = sheet.add_worksheet("MonitoredStocks", rows="1000", cols="10")
        ws.append_row(["Time", "Symbol", "Qty", "Buy Price", "CMP", "Trailing SL", "AI Exit Reason"])

    for record in records:
        ws.append_row(record)

def calculate_trailing_sl(buy_price, high_price, method='fixed', percent=3.0):
    if method == 'fixed':
        return round(high_price * (1 - percent/100), 2)
    elif method == 'ema':
        return round((buy_price + high_price) / 2 * 0.98, 2)
    else:
        return buy_price * 0.97

def mock_ai_exit_reason(symbol, cmp, sl, buy_price):
    if cmp < sl:
        return f"📉 Trailing SL Hit at ₹{cmp}"
    elif cmp < buy_price * 0.95:
        return f"🔻 Loss >5% from Buy Price"
    elif symbol in ["TCS", "LTIM"]:
        return "🔄 AI Exit: Reversal Pattern Detected"
    else:
        return ""

def monitor_loop():
    kite = init_kite()
    sheet = init_sheet()
    positions = fetch_positions(kite)

    msg = f"📈 *Falāh Monitor Started* – {datetime.now().strftime('%d %b %Y %H:%M')}\n"
    records = []
    for pos in positions:
        msg += f"🔹 {pos['tradingsymbol']} — Qty: {pos['quantity']} @ ₹{pos['average_price']}\n"
        records.append([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            pos['tradingsymbol'],
            pos['quantity'],
            pos['average_price'],
            "-", "-", "-"
        ])
    send_telegram(msg)
    log_to_sheet(sheet, records)

    tracked_highs = {pos['tradingsymbol']: pos['average_price'] for pos in positions}

    while True:
        try:
            if not is_market_open():
                print("🔕 Market closed — sleeping 5 min...")
                time.sleep(300)
                continue

            for pos in fetch_positions(kite):
                symbol = pos['tradingsymbol']
                buy_price = pos['average_price']
                qty = pos['quantity']

                ltp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]['last_price']
                tracked_highs[symbol] = max(tracked_highs.get(symbol, buy_price), ltp)
                trailing_sl = calculate_trailing_sl(buy_price, tracked_highs[symbol])

                reason = mock_ai_exit_reason(symbol, ltp, trailing_sl, buy_price)

                if reason:
                    msg = f"🚨 Exit Triggered: {symbol}\nQty: {qty}\nBuy: ₹{buy_price}\nCMP: ₹{ltp}\n{reason}"
                    send_telegram(msg)
                    sheet.worksheet("MonitoredStocks").append_row([
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        symbol, qty, buy_price, ltp, trailing_sl, reason
                    ])
                    print(f"[{symbol}] Exit reason: {reason}")

            time.sleep(900)

        except Exception as e:
            print("Error in monitor:", e)
            traceback.print_exc()
            send_telegram("❌ Falāh monitor error:\n" + str(e))
            time.sleep(300)

if __name__ == "__main__":
    monitor_loop()
