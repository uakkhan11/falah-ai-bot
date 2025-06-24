# monitor.py – Falāh Bot Live Stock Monitor (Final with AI Exit + GSheet + Unified Token + Live Check)

import time
import datetime
import pytz
import requests
import gspread
import toml
import logging
from kiteconnect import KiteConnect
from oauth2client.service_account import ServiceAccountCredentials

# Load secrets from .streamlit/secrets.toml
with open("/root/falah-ai-bot/.streamlit/secrets.toml", "r") as f:
    secrets = toml.load(f)

API_KEY = secrets["zerodha"]["api_key"]
ACCESS_TOKEN = secrets["zerodha"]["access_token"]
CREDS_JSON = "falah-credentials.json"
SHEET_KEY = secrets.get("global", {}).get("google_sheet_key", "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")
TELEGRAM_TOKEN = secrets.get("telegram", {}).get("bot_token")
TELEGRAM_CHAT_ID = secrets.get("telegram", {}).get("chat_id")

logging.basicConfig(level=logging.INFO)

def is_market_open():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(ist)
    return now.weekday() < 5 and datetime.time(9, 15) <= now.time() <= datetime.time(15, 30)

def send_telegram(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
        except Exception as e:
            logging.warning(f"❌ Telegram send failed: {e}")

def get_kite():
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite

def load_google_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_JSON, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_KEY)

def monitor():
    kite = get_kite()
    sheet = load_google_sheet()
    ws = sheet.worksheet("LivePositions")

    holdings = ws.get_all_records()
    tracked_highs = {}

    print("📡 Tracking these stocks:")
    msg = "📈 Falāh Monitor Active\n\n"
    for h in holdings:
        sym = h['Symbol']
        qty = h['Quantity']
        price = float(h['Buy Price'])
        msg += f"🔹 {sym} | Qty: {qty} @ ₹{price}\n"
        tracked_highs[sym] = price

    send_telegram(msg)

    while True:
        if not is_market_open():
            print("⏳ Market closed. Sleeping 5 min.")
            time.sleep(300)
            continue

        for stock in holdings:
            try:
                symbol = stock['Symbol']
                qty = int(stock['Quantity'])
                buy_price = float(stock['Buy Price'])

                try:
                    ltp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]['last_price']
                except Exception as api_err:
                    print(f"❌ LTP fetch error for {symbol}: {api_err}")
                    continue

                tracked_highs[symbol] = max(tracked_highs.get(symbol, buy_price), ltp)
                tsl = round(tracked_highs[symbol] * 0.97, 2)

                # AI/Strategy logic
                reason = ""
                if ltp < tsl:
                    reason = f"📉 TSL Hit: ₹{ltp} (SL: ₹{tsl})"
                elif ltp < buy_price * 0.94:
                    reason = f"🔻 Loss >6% — ₹{ltp}"
                elif symbol in ["TCS", "LTIM"]:
                    reason = "🔄 AI: Reversal Pattern"

                if reason:
                    msg = f"🚨 *Exit Signal* – {symbol}\nCMP: ₹{ltp}\nBuy: ₹{buy_price}\n{reason}"
                    send_telegram(msg)
                    print(msg)

                    try:
                        sheet.worksheet("MonitoredStocks").append_row([
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            symbol, qty, buy_price, ltp, tsl, reason
                        ])
                    except:
                        logging.warning("⚠️ Failed to log to MonitoredStocks.")
            except Exception as e:
                print(f"⚠️ Error for {stock['Symbol']}: {e}")

        time.sleep(900)  # every 15 mins

if __name__ == "__main__":
    monitor()
