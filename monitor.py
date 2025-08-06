# monitor.py

import os
import time
import json
import datetime
import argparse
import pandas as pd
import pandas_ta as ta
from kiteconnect import KiteConnect
from oauth2client.service_account import ServiceAccountCredentials
from trade_helpers import (compute_trailing_sl, send_telegram, log_trade_to_sheet, is_market_open)

# Load credentials
with open("/root/falah-ai-bot/secrets.json") as f:
    secrets = json.load(f)

API_KEY, ACCESS_TOKEN = secrets["zerodha"]["api_key"], secrets["zerodha"]["access_token"]
BOT_TOKEN, CHAT_ID = secrets["telegram"]["bot_token"], secrets["telegram"]["chat_id"]
SPREADSHEET_KEY = secrets["google"]["spreadsheet_key"]

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("falah-credentials.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SPREADSHEET_KEY)
log_sheet = sheet.worksheet("TradeLog")
monitor_sheet = sheet.worksheet("MonitoredStocks")

def monitor_positions(loop=True):
    send_telegram(BOT_TOKEN, CHAT_ID, "✅ Falāh Monitoring Started ✅")
    equity_peak = None
    while True:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        positions = kite.positions()["net"]
        holdings = kite.holdings()

        # Merge holdings + positions
        merged_positions = {}
        for h in holdings:
            qty = h["quantity"] + h["t1_quantity"]
            if qty > 0:
                merged_positions[h["tradingsymbol"]] = {
                    "quantity": qty,
                    "average_price": h["average_price"],
                    "last_price": h["last_price"],
                    "exchange": h["exchange"]
                }

        for p in positions:
            if p["quantity"] != 0 and p["tradingsymbol"] not in merged_positions:
                merged_positions[p["tradingsymbol"]] = {
                    "quantity": p["quantity"],
                    "average_price": p["average_price"],
                    "last_price": p["last_price"],
                    "exchange": p["exchange"]
                }

        if not merged_positions:
            print(f"{timestamp} ❌ No active positions.")
            if not loop: break
            time.sleep(900); continue

        portfolio_value = sum(p["last_price"] * p["quantity"] for p in merged_positions.values())
        if equity_peak is None: equity_peak = portfolio_value
        if portfolio_value > equity_peak: equity_peak = portfolio_value
        drawdown_pct = (equity_peak - portfolio_value) / equity_peak * 100
        print(f"\n{timestamp} ✅ Monitoring {len(merged_positions)} stocks | Drawdown: {drawdown_pct:.2f}%")

        # Max Drawdown Full Exit
        if drawdown_pct >= 7:
            send_telegram(BOT_TOKEN, CHAT_ID, f"❌ Drawdown {drawdown_pct:.2f}% triggered. Full Exit initiated.")
            for symbol, pos in merged_positions.items():
                kite.place_order(
                    variety=kite.VARIETY_REGULAR, exchange=pos["exchange"],
                    tradingsymbol=symbol, transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=pos["quantity"], order_type=kite.ORDER_TYPE_MARKET, product=kite.PRODUCT_CNC
                )
            break

        # Individual Monitoring
        for symbol, pos in merged_positions.items():
            qty, avg_price, exchange = pos["quantity"], pos["average_price"], pos["exchange"]
            try:
                ltp = kite.ltp(f"{exchange}:{symbol}")[f"{exchange}:{symbol}"]["last_price"]
            except:
                ltp = pos["last_price"]

            print(f"\n▶️ {symbol} | Qty:{qty} | LTP:{ltp:.2f}")

            # Load historical data
            try:
                df = pd.read_csv(f"/root/falah-ai-bot/historical_data/{symbol}.csv")
            except FileNotFoundError:
                print(f"⚠️ No data for {symbol}, skipping."); continue

            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            trailing_sl_atr = ltp - 1.5 * df["atr"].iloc[-1]
            trailing_sl_recent = df["low"].rolling(7).min().iloc[-1]
            trailing_sl = max(trailing_sl_atr, trailing_sl_recent)

            print(f"🔹 Trailing SL: {trailing_sl:.2f} | ATR SL: {trailing_sl_atr:.2f} | Recent Low SL: {trailing_sl_recent:.2f}")

            ai_score, reasons = 0, []
            if ltp < avg_price * 0.98:
                ai_score += 25; reasons.append("Fixed SL breach (-2%)")
            if ltp < trailing_sl:
                ai_score += 20; reasons.append("Trailing SL breached")
            if ltp >= avg_price * 1.12:
                ai_score += 10; reasons.append("Profit >=12% hit")

            print(f"⚡ AI Score: {ai_score} | Reasons: {', '.join(reasons) or 'Holding'}")

            # Exit Logic
            exit_qty, exit_reason = 0, ""
            if ai_score >= 30:
                exit_qty, exit_reason = qty, "Full Exit"
            elif 15 <= ai_score < 30:
                exit_qty, exit_reason = qty // 2, "Partial Exit"

            if exit_qty > 0 and is_market_open():
                kite.place_order(
                    variety=kite.VARIETY_REGULAR, exchange=exchange,
                    tradingsymbol=symbol, transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=exit_qty, order_type=kite.ORDER_TYPE_MARKET, product=kite.PRODUCT_CNC
                )
                send_telegram(BOT_TOKEN, CHAT_ID,
                    f"⚠️ {exit_reason}: {symbol} {exit_qty} qty exited | PnL ₹{(ltp - avg_price) * exit_qty:.2f}"
                )

        if not loop: break
        time.sleep(900)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run only once.")
    args = parser.parse_args()
    monitor_positions(loop=not args.once)
