import os
import json
import time
from datetime import datetime
import pandas as pd
from kiteconnect import KiteConnect
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from pytz import timezone
from trade_helpers import (
    calculate_atr,
    compute_trailing_sl,
    send_telegram,
    log_trade_to_sheet,
    is_market_open
)

# 🟢 Load credentials
with open("/root/falah-ai-bot/secrets.json", "r") as f:
    secrets = json.load(f)

API_KEY = secrets["zerodha"]["api_key"]
ACCESS_TOKEN = secrets["zerodha"]["access_token"]
BOT_TOKEN = secrets["telegram"]["bot_token"]
CHAT_ID = secrets["telegram"]["chat_id"]
SPREADSHEET_KEY = secrets["google"]["spreadsheet_key"]

# 🟢 Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# 🟢 Google Sheets Auth
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("falah-credentials.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SPREADSHEET_KEY)
log_sheet = sheet.worksheet("TradeLog")

# 🟢 Telegram Helper
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"[Telegram Error] {e}")

# 🟢 Trailing SL Logic
def trailing_stop_loop():
    india = timezone("Asia/Kolkata")
    while True:
        holdings = kite.holdings()
        timestamp = datetime.now(india).strftime("%Y-%m-%d %H:%M:%S")

        if not holdings:
            print("⚠️ No holdings found. Sleeping 10 min...")
            time.sleep(600)
            continue

        print(f"\n🕒 {timestamp} - Checking Trailing SLs for {len(holdings)} positions.")

        for pos in holdings:
            symbol = pos["tradingsymbol"]
            qty = pos["quantity"]
            avg_price = pos["average_price"]

            ltp_data = kite.ltp(f"NSE:{symbol}")
            ltp = ltp_data[f"NSE:{symbol}"]["last_price"]

            # Compute dynamic trailing SL: e.g., highest close minus 1.5 * ATR
            hist = kite.historical_data(
                pos["instrument_token"],
                datetime.now(india) - pd.Timedelta(days=30),
                datetime.now(india),
                "day"
            )
            df = pd.DataFrame(hist)
            df.columns = [c.lower() for c in df.columns]
            df = df.dropna()

            atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
            highest_close = df["close"].rolling(10).max().iloc[-1]
            trailing_sl = round(highest_close - atr * 1.5, 2)

            print(f"🔍 {symbol}: LTP ₹{ltp}, Trailing SL ₹{trailing_sl}")

            if ltp < trailing_sl:
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
                    msg = (
                        f"⚠️ <b>Trailing SL Hit - Exited {symbol}</b>\n"
                        f"LTP ₹{ltp}\nTrailing SL ₹{trailing_sl}\nQty {qty}"
                    )
                    send_telegram(msg)
                    log_sheet.append_row([
                        timestamp, symbol, "SELL", qty, ltp, "Trailing SL Hit", f"Trailing SL {trailing_sl}"
                    ])
                    print(f"✅ Exited {symbol} and logged.")
                except Exception as e:
                    print(f"❌ Error exiting {symbol}: {e}")
        print("✅ Cycle complete. Sleeping 10 min.\n")
        time.sleep(600)

# 🟢 Entry point
if __name__ == "__main__":
    trailing_stop_loop()
