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
import csv
from kiteconnect import KiteConnect
from oauth2client.service_account import ServiceAccountCredentials
from pytz import timezone

from indicators import (
    detect_rsi_ema_signals,
    detect_3green_days,
    detect_macd_cross,
    detect_darvas_box
)
from trade_helpers import (
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

# 🟢 Load Nifty data
def load_historical_df(symbol):
    path = f"/root/falah-ai-bot/historical_data/{symbol}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    required = {"date", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        print(f"⚠️ Missing columns for {symbol}: {required - set(df.columns)}")
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

nifty_df = load_historical_df("NIFTY")
if nifty_df is None:
    raise Exception("NIFTY data required.")

def detect_market_regime(df):
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    return "TREND" if adx["ADX_14"].iloc[-1] > 25 else "RANGE"

# 🟢 Local backup log file
BACKUP_LOG_PATH = "/root/falah-ai-bot/monitor_log.csv"
if not os.path.exists(BACKUP_LOG_PATH):
    with open(BACKUP_LOG_PATH, "w") as f:
        csv.writer(f).writerow([
            "Timestamp", "Symbol", "Quantity", "AvgPrice", "LTP",
            "TrailingSL", "RelStrength", "Regime", "AIScore", "Reasons"
        ])

# 🟢 Main loop
def monitor_positions(loop=True):
    send_telegram("✅ <b>Falāh Monitoring Started</b>\nMonitoring CNC positions.")
    last_heartbeat = datetime.datetime.now()

    while True:
        now = datetime.datetime.now()
        if (now - last_heartbeat).seconds >= 3600:
            send_telegram("💓 Heartbeat: Monitoring script alive.")
            last_heartbeat = now

        print("\n==============================")
        print(f"🕒 {now.strftime('%Y-%m-%d %H:%M:%S')} - Scanning positions...")
        holdings = kite.holdings()

        if not holdings:
            print("⚠️ No holdings found.")
            time.sleep(900)
            continue

        all_rows = []
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        for pos in holdings:
            symbol = pos["tradingsymbol"]
            qty = pos["quantity"]
            avg_price = pos["average_price"]
            exchange = pos["exchange"]

            ltp = kite.ltp(f"{exchange}:{symbol}")[f"{exchange}:{symbol}"]["last_price"]
            print(f"\n🔍 {symbol}: LTP = ₹{ltp}")

            df = load_historical_df(symbol)
            if df is None:
                continue

            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            rsi_series = ta.rsi(df["close"], length=14)

            regime = detect_market_regime(df)
            rel_strength = df["close"].iloc[-1] / nifty_df["close"].iloc[-1]
            rsi_latest = rsi_series.iloc[-1]
            rsi_percentile = (rsi_latest - rsi_series.min()) / (rsi_series.max() - rsi_series.min())

            adx_val = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"].iloc[-1]

            # Dynamic trailing stop (suggestion #1)
            atr_multiplier = 1.0 if adx_val > 35 else 1.5
            trailing_sl = compute_trailing_sl(df, atr_multiplier=atr_multiplier)

            # Target price logic (suggestion #2)
            target_price = round(avg_price + (avg_price - trailing_sl) * 3, 2)

            rsi_ema = detect_rsi_ema_signals(df)
            macd_cross = detect_macd_cross(df)
            three_green = detect_3green_days(df)
            darvas, _ = detect_darvas_box(df)

            ai_score = 0
            reasons = []

            if ltp < avg_price * 0.98:
                ai_score += 15
                reasons.append("Loss >2%")

            if ltp >= target_price:
                ai_score += 100
                reasons.append("Target price hit")

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

            print(f"✅ AI Score: {ai_score} | {', '.join(reasons)}")

            # Decide exit
            exit_qty = qty if ai_score >= 70 else (int(qty * 0.5) if ai_score >= 50 else 0)

            all_rows.append([
                timestamp, symbol, qty, avg_price, ltp,
                trailing_sl, round(rel_strength, 4), regime,
                ai_score, ", ".join(reasons)
            ])

            # Local backup log (suggestion #5)
            with open(BACKUP_LOG_PATH, "a") as f:
                csv.writer(f).writerow([
                    timestamp, symbol, qty, avg_price, ltp,
                    trailing_sl, round(rel_strength, 4), regime,
                    ai_score, "; ".join(reasons)
                ])

            # Exit order
            if exit_qty > 0:
                if is_market_open():
                    retries = 3
                    while retries > 0:
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
                            log_trade_to_sheet(
                                log_sheet, timestamp, symbol, exit_qty, ltp,
                                "Exit Triggered", ", ".join(reasons)
                            )
                            print(f"✅ Exit order placed for {symbol}.")
                            break
                        except Exception as e:
                            retries -= 1
                            print(f"Retrying ({3-retries}/3)... {e}")
                            time.sleep(3)
                    else:
                        send_telegram(f"❌ Failed exit after retries for {symbol}")
                else:
                    send_telegram(f"⚠️ Exit signal for {symbol}, but market closed.")

        if all_rows:
            monitor_sheet.append_rows(all_rows, value_input_option="USER_ENTERED")

        print("✅ Monitoring cycle done.\n")
        if not loop:
            break
        time.sleep(900)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one monitoring cycle")
    args = parser.parse_args()
    monitor_positions(loop=not args.once)
