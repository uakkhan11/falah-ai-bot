# monitor.py

import os
import time
import json
import datetime
import argparse
import pandas as pd
import requests
import pandas_ta as ta
import gspread
from kiteconnect import KiteConnect
from oauth2client.service_account import ServiceAccountCredentials
from pytz import timezone

from indicators import (
    detect_rsi_ema_signals,
    detect_3green_days,
    detect_macd_cross,
    detect_darvas_box
)

# 🟢 Load credentials
with open("/root/falah-ai-bot/secrets.json", "r") as f:
    secrets = json.load(f)

API_KEY = secrets["zerodha"]["api_key"]
ACCESS_TOKEN = secrets["zerodha"]["access_token"]
BOT_TOKEN = secrets["telegram"]["bot_token"]
CHAT_ID = secrets["telegram"]["chat_id"]
SPREADSHEET_KEY = secrets["google"]["spreadsheet_key"]

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

# 🟢 Helper: Send Telegram message
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"[Telegram Error] {e}")

# 🟢 Helper: Market hours check
def is_market_open():
    india = timezone("Asia/Kolkata")
    now = datetime.datetime.now(india)
    return (
        now.weekday() < 5 and
        now.hour >= 9 and
        (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
    )

# 🟢 Helper: Load historical OHLC
def load_historical_df(symbol):
    path = f"/root/falah-ai-bot/historical_data/{symbol}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Normalize column names
    # df.columns = [c.strip().lower() for c in df.columns]
    # Ensure required columns
    required = {"date", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        print(f"⚠️ Historical data for {symbol} missing columns: {required - set(df.columns)}")
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# 🟢 Load Nifty for relative strength
nifty_df = load_historical_df("NIFTY")
if nifty_df is None:
    raise Exception("Nifty historical data required (NIFTY.csv).")

# 🟢 Market regime detection
def detect_market_regime(df):
    adx = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=14)
    if adx["ADX_14"].iloc[-1] > 25:
        return "TREND"
    else:
        return "RANGE"

# 🟢 Main monitoring loop
def monitor_positions(loop=True):
    send_telegram("✅ <b>Falāh Monitoring Started</b>\nTracking all CNC holdings...")

    while True:
        print("\n==============================")
        print(f"🕒 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Scanning positions...")
        holdings = kite.holdings()

        if not holdings:
            print("⚠️ No CNC holdings found.")
        else:
            print(f"✅ Found {len(holdings)} holdings.")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_rows = []

        for pos in holdings:
            symbol = pos["tradingsymbol"]
            qty = pos["quantity"]
            avg_price = pos["average_price"]
            exchange = pos["exchange"]

            ltp_data = kite.ltp(f"{exchange}:{symbol}")
            ltp = ltp_data[f"{exchange}:{symbol}"]["last_price"]
            print(f"\n🔍 {symbol}: LTP = ₹{ltp}")

            df = load_historical_df(symbol)
            if df is None:
                print(f"⚠️ No historical data for {symbol}. Skipping.")
                continue
                
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            
            # Indicators
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            rsi_series = ta.rsi(df["close"], length=14)

            regime = detect_market_regime(df)
            rel_strength = df["close"].iloc[-1] / nifty_df["close"].iloc[-1]
            rsi_latest = rsi_series.iloc[-1]
            rsi_percentile = (rsi_latest - rsi_series.min()) / (rsi_series.max() - rsi_series.min())

            rsi_ema = detect_rsi_ema_signals(df)
            macd_cross = detect_macd_cross(df)
            three_green = detect_3green_days(df)
            darvas, _ = detect_darvas_box(df)

            atr_value = df["atr"].iloc[-1]
            trailing_sl = round(ltp - atr_value * (1.5 if rsi_percentile < 0.5 else 1.0), 2)

            # AI scoring
            ai_score = 0
            reasons = []

            if ltp < avg_price * 0.98:
                ai_score += 15
                reasons.append("Loss >2%")

            if regime == "RANGE":
                ai_score += 10
                reasons.append("Market RANGE")

            if ltp < df["vwap"].iloc[-1]:
                ai_score += 10
                reasons.append("Below VWAP")

            if rel_strength < 0.98:
                ai_score += 10
                reasons.append("Weak relative strength")

            if rsi_percentile < 0.2:
                ai_score += 15
                reasons.append("RSI oversold")
            if rsi_percentile > 0.8:
                ai_score += 15
                reasons.append("RSI overbought")

            weights = {"rsi_ema": 0.3, "macd_cross": 0.3, "three_green": 0.1, "darvas_box": 0.2}
            if not rsi_ema:
                ai_score += 100 * weights["rsi_ema"]
                reasons.append("RSI/EMA weak")
            if not macd_cross:
                ai_score += 100 * weights["macd_cross"]
                reasons.append("MACD no cross")
            if not three_green:
                ai_score += 100 * weights["three_green"]
                reasons.append("No 3 green days")
            if not darvas:
                ai_score += 100 * weights["darvas_box"]
                reasons.append("Darvas Box absent")
            if ltp < trailing_sl:
                ai_score += 10
                reasons.append("Trailing SL breached")

            print(f"✅ AI Score: {ai_score} | {', '.join(reasons) if reasons else 'Holding'}")

            # Exit decision
            exit_qty = 0
            if ai_score >= 70:
                exit_qty = qty
            elif ai_score >= 50:
                exit_qty = int(qty * 0.5)

            all_rows.append([
                timestamp, symbol, qty, avg_price, ltp,
                trailing_sl, round(rel_strength, 4),
                regime, ai_score,
                ", ".join(reasons) if reasons else "Holding"
            ])

            if exit_qty > 0:
                if is_market_open():
                    try:
                        kite.place_order(
                            variety=kite.VARIETY_REGULAR,
                            exchange=exchange,
                            tradingsymbol=symbol,
                            transaction_type=kite.TRANSACTION_TYPE_SELL,
                            quantity=exit_qty,
                            order_type=kite.ORDER_TYPE_MARKET,
                            product=kite.PRODUCT_CNC
                        )
                        send_telegram(f"⚠️ <b>Exit Triggered</b>\n{symbol}\nQty:{exit_qty}\nLTP:{ltp}\nReasons:{', '.join(reasons)}")
                        log_sheet.append_row([
                            timestamp, symbol, "SELL", exit_qty, ltp, "Exit Triggered", ", ".join(reasons)
                        ])
                        print(f"✅ Exit order placed for {symbol}.")
                    except Exception as e:
                        send_telegram(f"❌ Exit error for {symbol}: {e}")
                        print(f"❌ Error placing exit order: {e}")
                else:
                    send_telegram(
                        f"⚠️ <b>Exit Signal Generated but Market Closed</b>\n"
                        f"{symbol}\nQty:{exit_qty}\nLTP:{ltp}\nReasons:{', '.join(reasons)}"
                    )
                    print(f"⚠️ Market closed. Exit pending for {symbol}.")

        if all_rows:
            monitor_sheet.append_rows(all_rows, value_input_option="USER_ENTERED")

        print("✅ Monitoring cycle completed.\n")
        if not loop:
            break
        time.sleep(900)

# 🟢 Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one monitoring cycle and exit")
    args = parser.parse_args()
    monitor_positions(loop=not args.once)
