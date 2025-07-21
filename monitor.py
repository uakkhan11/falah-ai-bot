# monitor.py

import os
import time
import json
import datetime
import argparse
import pandas as pd
import pandas_ta as ta
import gspread
from kiteconnect import KiteConnect
from oauth2client.service_account import ServiceAccountCredentials

from trade_helpers import (
    compute_trailing_sl,
    send_telegram,
    log_trade_to_sheet,
    is_market_open
)

# ✅ Load credentials
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
    send_telegram(BOT_TOKEN, CHAT_ID, "✅ Falāh Monitoring Started")
    equity_peak = None

    while True:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        positions = kite.positions()["net"]
        holdings = kite.holdings()
        merged_positions = []

        symbols_in_holdings = [h["tradingsymbol"] for h in holdings]

        # ✅ Holdings (T2 + T1)
        for h in holdings:
            qty = h["quantity"] + h["t1_quantity"]
            if qty > 0:
                merged_positions.append({
                    "tradingsymbol": h["tradingsymbol"],
                    "quantity": qty,
                    "average_price": h["average_price"],
                    "last_price": h["last_price"],
                    "exchange": h["exchange"]
                })

        # ✅ Positions not in holdings
        for p in positions:
            if p["quantity"] != 0 and p["tradingsymbol"] not in symbols_in_holdings and p["product"] != "MIS":
                merged_positions.append({
                    "tradingsymbol": p["tradingsymbol"],
                    "quantity": p["quantity"],
                    "average_price": p["average_price"],
                    "last_price": p["last_price"],
                    "exchange": p["exchange"]
                })

        if not merged_positions:
            print(f"{timestamp} ❌ No active positions.")
            if not loop: break
            time.sleep(900); continue

        portfolio_value = sum(p["last_price"] * p["quantity"] for p in merged_positions)
        if equity_peak is None: equity_peak = portfolio_value
        drawdown_pct = (equity_peak - portfolio_value) / equity_peak * 100
        print(f"\n{timestamp} ✅ Monitoring {len(merged_positions)} stocks | Drawdown: {drawdown_pct:.2f}%")

        all_rows = []

        if drawdown_pct >= 7:
            send_telegram(BOT_TOKEN, CHAT_ID, f"❌ Drawdown {drawdown_pct:.2f}% — Full Exit.")
            for pos in merged_positions:
                kite.place_order(
                    variety=kite.VARIETY_REGULAR, exchange=pos["exchange"],
                    tradingsymbol=pos["tradingsymbol"], transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=pos["quantity"], order_type=kite.ORDER_TYPE_MARKET, product=kite.PRODUCT_CNC
                )
            break

        for pos in merged_positions:
            symbol, qty, avg_price = pos["tradingsymbol"], pos["quantity"], pos["average_price"]
            ltp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]
            print(f"\n▶️ {symbol} | Qty:{qty} | LTP:{ltp:.2f}")

            try:
                df = pd.read_csv(f"/root/falah-ai-bot/historical_data/{symbol}.csv")
            except FileNotFoundError:
                print(f"⚠️ No data for {symbol}, skipping."); continue

            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            trailing_sl = compute_trailing_sl(df)

            ai_score, reasons = 0, []

            # ✅ Fixed SL
            if ltp < avg_price * 0.98:
                ai_score += 25
                reasons.append("Fixed SL 2% hit")

            # ✅ Trailing SL logic
            if ltp < trailing_sl:
                ai_score += 20
                reasons.append("Trailing SL breached")

            # ✅ Profit Booking
            if ltp >= avg_price * 1.12:
                ai_score += 10
                reasons.append("Profit >12% secured")

            print(f"SL:{trailing_sl:.2f} | AI:{ai_score} | Reasons: {', '.join(reasons)}")

            exit_qty, exit_reason = 0, ""
            if ai_score >= 30:
                exit_qty, exit_reason = qty, "Full Exit Triggered"
            elif 15 <= ai_score < 30:
                exit_qty, exit_reason = qty // 2, "Partial Exit Triggered"

            pnl = (ltp - avg_price) * exit_qty if exit_qty > 0 else 0

            all_rows.append([
                timestamp, symbol, qty, avg_price, ltp,
                trailing_sl, ai_score, ", ".join(reasons) or "Holding", pnl
            ])

            if exit_qty > 0 and is_market_open():
                kite.place_order(
                    variety=kite.VARIETY_REGULAR, exchange="NSE",
                    tradingsymbol=symbol, transaction_type=kite.TRANSACTION_TYPE_SELL",
                    quantity=exit_qty, order_type=kite.ORDER_TYPE_MARKET, product=kite.PRODUCT_CNC
                )
                send_telegram(BOT_TOKEN, CHAT_ID,
                    f"⚠️ {exit_reason}: {symbol} {exit_qty} qty | PnL ₹{pnl:.2f}")
                log_trade_to_sheet(
                    log_sheet, timestamp, symbol, exit_qty, avg_price, ltp,
                    "", "", "", ai_score, "SELL", exit_reason, pnl, 1 if pnl >= 0 else 0
                )

        if all_rows:
            monitor_sheet.append_rows(all_rows, value_input_option="USER_ENTERED")

        if not loop: break
        time.sleep(900)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run only once.")
    args = parser.parse_args()
    monitor_positions(loop=not args.once)
