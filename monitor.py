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

# 🟢 Load Nifty for relative strength
nifty_df = pd.read_csv("/root/falah-ai-bot/historical_data/NIFTY.csv")
nifty_df["date"] = pd.to_datetime(nifty_df["date"])
nifty_df = nifty_df.sort_values("date").reset_index(drop=True)

# 🟢 Main monitoring loop
def monitor_positions(loop=True):
    send_telegram("✅ <b>Falāh Monitoring Started</b>")

    # Track peak equity for drawdown
    equity_peak = None

    while True:
        print("\n==============================")
        holdings = kite.holdings()

        if not holdings:
            print("⚠️ No CNC holdings.")
        else:
            print(f"✅ Found {len(holdings)} holdings.")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_rows = []

        # Compute portfolio value
        portfolio_value = sum(h["last_price"] * h["quantity"] for h in holdings)
        if equity_peak is None:
            equity_peak = portfolio_value

        drawdown_pct = (equity_peak - portfolio_value) / equity_peak * 100
        print(f"Current Drawdown: {drawdown_pct:.2f}%")

        if drawdown_pct >= 7:
            send_telegram(f"❌ <b>Drawdown Limit Breached ({drawdown_pct:.2f}%)</b>. Exiting all positions.")
            for pos in holdings:
                kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=pos["exchange"],
                    tradingsymbol=pos["tradingsymbol"],
                    transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=pos["quantity"],
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_CNC
                )
                log_trade_to_sheet(
                    log_sheet, timestamp, pos["tradingsymbol"], pos["quantity"], pos["last_price"],
                    "Drawdown Exit", ""
                )
            break

        for pos in holdings:
            symbol = pos["tradingsymbol"]
            qty = pos["quantity"]
            avg_price = pos["average_price"]

            ltp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]
            df = pd.read_csv(f"/root/falah-ai-bot/historical_data/{symbol}.csv")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            trailing_sl = compute_trailing_sl(df)

            ai_score = 0
            reasons = []

            if ltp < avg_price * 0.98:
                ai_score += 15
                reasons.append("Loss >2%")

            if ltp < trailing_sl:
                ai_score += 20
                reasons.append("Trailing SL breached")

            # Dynamic trailing SL adjustment: move up if price advances
            new_sl = max(trailing_sl, avg_price + df["atr"].iloc[-1])

            print(f"✅ {symbol} AI Score: {ai_score}")

            exit_qty = qty if ai_score >= 20 else 0

            all_rows.append([
                timestamp, symbol, qty, avg_price, ltp, trailing_sl,
                "", ai_score, ", ".join(reasons) if reasons else "Holding"
            ])

            if exit_qty > 0:
                if is_market_open():
                    kite.place_order(
                        variety=kite.VARIETY_REGULAR,
                        exchange="NSE",
                        tradingsymbol=symbol,
                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                        quantity=exit_qty,
                        order_type=kite.ORDER_TYPE_MARKET,
                        product=kite.PRODUCT_CNC
                    )
                    send_telegram(f"⚠️ Exit {symbol} Qty:{exit_qty} Reason:{', '.join(reasons)}")
                    log_trade_to_sheet(
                        log_sheet, timestamp, symbol, exit_qty, ltp,
                        "Auto Exit", ", ".join(reasons)
                    )

        if all_rows:
            monitor_sheet.append_rows(all_rows, value_input_option="USER_ENTERED")

        if not loop:
            break

        time.sleep(900)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one monitoring cycle")
    args = parser.parse_args()
    monitor_positions(loop=not args.once)
