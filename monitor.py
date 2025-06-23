import time
import pandas as pd
import numpy as np
import requests
import talib
from kiteconnect import KiteConnect
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz
import logging

# === CONFIGURATION ===
API_KEY = "ijzeuwuylr3g0kug"
ACCESS_TOKEN = "HdiGzilCwOT8yZqCreno4mb7FyJwxMOy"
CREDS_FILE = "falah-credentials.json"
SHEET_KEY = "1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c"
TELEGRAM_TOKEN = "7763450358:AAH32bWYyu_hXR6l-UaVMaarFGZ4YFOv6q8"
TELEGRAM_CHAT_ID = "6784139148"
INTERVAL = 15 * 60  # 15 minutes
IST = pytz.timezone("Asia/Kolkata")


# === SETUP LOGGING ===
logging.basicConfig(filename="monitor.log", level=logging.INFO)

# === KITE INIT ===
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# === GOOGLE SHEET INIT ===
def init_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_KEY)

sheet = init_sheet()


# === TELEGRAM ALERT ===
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(url, data=payload)
    except:
        logging.error("Telegram send failed.")


# === NEWS PLACEHOLDER ===
def has_negative_news(stock):
    # Placeholder mock logic
    # You can integrate NewsAPI or tag logic here
    return False  # Return True to block exit

# === TRAILING SL ENGINE ===
def compute_trailing_sl(df, entry_price):
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)[-1]
    ema = talib.EMA(df['close'], timeperiod=20)[-1]
    highest = df['close'].cummax().iloc[-1]
    
    sl_price = min(
        highest * 0.97,        # 3% below peak
        highest - atr,         # ATR based
        ema                    # EMA support
    )
    return round(sl_price, 2)

# === EXIT STRATEGY DECISION ===
def should_exit(stock, ltp, trailing_sl, df_15m, df_1h):
    if ltp < trailing_sl:
        logging.info(f"{stock} below trailing SL.")
        if has_negative_news(stock):
            return True, "SL hit + negative news"

        # Reverse pattern check (mock)
        if "hammer" in detect_patterns(df_15m) or "doji" in detect_patterns(df_1h):
            return False, "Hold: reversal pattern detected"

        return True, "SL hit: confirmed by multi-timeframe"
    return False, "Hold: still above trailing SL"

# === PATTERN SCANNER ===
def detect_patterns(df):
    result = []
    if talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])[-1] != 0:
        result.append("hammer")
    if talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])[-1] != 0:
        result.append("doji")
    if talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])[-1] != 0:
        result.append("engulfing")
    return result

# === LOAD LIVE CNC POSITIONS ===
def get_live_positions():
    positions = kite.positions()
    holdings = positions['day'] + positions['net']
    cnc = [p for p in holdings if p['product'] == 'CNC' and p['quantity'] > 0]
    return cnc

# === GET HISTORICAL DATA ===
def fetch_ohlc(symbol, interval, days):
    try:
        instrument_token = kite.ltp(f'NSE:{symbol}')[f'NSE:{symbol}']['instrument_token']
        from_date = (datetime.now() - pd.Timedelta(days=days)).date()
        to_date = datetime.now().date()
        data = kite.historical_data(instrument_token, from_date, to_date, interval)
        return pd.DataFrame(data)
    except Exception as e:
        logging.warning(f"{symbol}: historical fetch failed - {e}")
        return pd.DataFrame()

# === ALREADY SOLD CHECK ===
def is_already_sold(symbol):
    try:
        ws = sheet.worksheet("SellLog")
        sold_list = ws.col_values(1)
        return symbol in sold_list
    except:
        return False

# === LOG SELL ===
def log_sell(symbol, ltp, reason):
    try:
        ws = sheet.worksheet("SellLog")
        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([symbol, ltp, reason, now])
    except:
        logging.warning(f"{symbol} sell log failed.")

# === MAIN LOOP ===
def check_positions_once():
    live = get_live_positions()
    for p in live:
        symbol = p['tradingsymbol']
        if is_already_sold(symbol):
            continue

        df = fetch_ohlc(symbol, "5minute", 5)
        df_15 = fetch_ohlc(symbol, "15minute", 7)
        df_1h = fetch_ohlc(symbol, "60minute", 10)

        if df.empty or df_15.empty or df_1h.empty:
            continue

        trailing_sl = compute_trailing_sl(df, p['average_price'])
        ltp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]['last_price']
        exit_now, reason = should_exit(symbol, ltp, trailing_sl, df_15, df_1h)

        if exit_now:
            try:
                order = kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NSE,
                    tradingsymbol=symbol,
                    transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=p['quantity'],
                    product=kite.PRODUCT_CNC,
                    order_type=kite.ORDER_TYPE_MARKET
                )
                log_sell(symbol, ltp, reason)
                send_telegram(f"🚨 Sold {symbol} at {ltp} – Reason: {reason}")
                logging.info(f"Sold {symbol}: {reason}")
            except Exception as e:
                logging.error(f"Sell failed for {symbol}: {e}")
        else:
            logging.info(f"{symbol} held: {reason}")

# === SCHEDULER LOOP ===
if __name__ == "__main__":
    while True:
        try:
            logging.info("🟢 Monitoring started...")
            check_positions_once()
        except Exception as e:
            logging.error(f"Monitoring error: {e}")
        time.sleep(INTERVAL)
