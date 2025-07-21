# Saving the updated monitor.py as a full working file

monitor_py_code = """
# monitor.py (updated with refined SL logic, partial exits, P&L reporting)

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

with open("/root/falah-ai-bot/secrets.json", "r") as f:
    secrets = json.load(f)

API_KEY = secrets["zerodha"]["api_key"]
ACCESS_TOKEN = secrets["zerodha"]["access_token"]
BOT_TOKEN = secrets["telegram"]["bot_token"]
CHAT_ID = secrets["telegram"]["chat_id"]
SPREADSHEET_KEY = secrets["google"]["spreadsheet_key"]

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("falah-credentials.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SPREADSHEET_KEY)
log_sheet = sheet.worksheet("TradeLog")
monitor_sheet = sheet.worksheet("MonitoredStocks")

nifty_df = pd.read_csv("/root/falah-ai-bot/historical_data/NIFTY.csv")
nifty_df.rename(columns={"Date": "date"}, inplace=True)
nifty_df["date"] = pd.to_datetime(nifty_df["date"])
nifty_df = nifty_df.sort_values("date").reset_index(drop=True)

def monitor_positions(loop=True):
    send_telegram(BOT_TOKEN, CHAT_ID, "✅ <b>Falāh Monitoring Started</b>")
    equity_peak = None

    while True:
        print("\\n==============================")
        positions = kite.positions()["net"]
        holdings = kite.holdings()
        merged_positions = []

        symbols_in_holdings = [x["tradingsymbol"] for x in holdings]

        for h in holdings:
            effective_qty = h["quantity"] + h["t1_quantity"]
            if effective_qty == 0:
                continue
            merged_positions.append({
                "tradingsymbol": h["tradingsymbol"],
                "quantity": effective_qty,
                "average_price": h["average_price"],
                "last_price": h["last_price"],
                "exchange": h["exchange"],
                "source": f"holdings (T2:{h['quantity']} T1:{h['t1_quantity']})"
            })

        for p in positions:
            if p["quantity"] == 0 or p["product"] == "MIS":
                continue
            if p["tradingsymbol"] in symbols_in_holdings:
                continue
            merged_positions.append({
                "tradingsymbol": p["tradingsymbol"],
                "quantity": p["quantity"],
                "average_price": p["average_price"],
                "last_price": p["last_price"],
                "exchange": p["exchange"],
                "source": "positions"
            })

        if not merged_positions:
            print("⚠️ No CNC holdings or positions.")
        else:
            print(f"✅ Found {len(merged_positions)} positions.")
            for pos in merged_positions:
                print(f"▶️ Monitoring: {pos['tradingsymbol']} | Qty: {pos['quantity']} | Source: {pos['source']}")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_rows = []

        portfolio_value = sum(p["last_price"] * p["quantity"] for p in merged_positions)
        if equity_peak is None:
            equity_peak = portfolio_value

        drawdown_pct = (equity_peak - portfolio_value) / equity_peak * 100
        print(f"Current Drawdown: {drawdown_pct:.2f}%")

        if drawdown_pct >= 7:
            send_telegram(BOT_TOKEN, CHAT_ID, f"❌ <b>Drawdown Limit Breached ({drawdown_pct:.2f}%)</b>. Exiting all positions.")
            for pos in merged_positions:
                kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=pos["exchange"],
                    tradingsymbol=pos["tradingsymbol"],
                    transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=pos["quantity"],
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_CNC
                )
                pnl = (pos["last_price"] - pos["average_price"]) * pos["quantity"]
                outcome = 1 if pnl > 0 else 0
                log_trade_to_sheet(log_sheet, timestamp, pos["tradingsymbol"], pos["quantity"], pos["average_price"], pos["last_price"], "", "", "", "", "SELL", "Drawdown Exit", pnl, outcome)
            break

        for pos in merged_positions:
            symbol = pos["tradingsymbol"]
            qty = pos["quantity"]
            avg_price = pos["average_price"]

            ltp = kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["last_price"]
            try:
                df = pd.read_csv(f"/root/falah-ai-bot/historical_data/{symbol}.csv")
            except FileNotFoundError:
                print(f"⚠️ No historical data for {symbol}. Skipping this position.")
                continue

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            trailing_sl = compute_trailing_sl(df)

            ai_score = 0
            reasons = []

            if ltp < avg_price * 0.96:
                ai_score += 20
                reasons.append("Fixed 4% SL Hit")

            if ltp < trailing_sl:
                ai_score += 15
                reasons.append("Trailing SL Breached")

            if ltp > avg_price * 1.15:
                ai_score += 10
                reasons.append("Profit booking >15%")

            new_sl = max(trailing_sl, avg_price + df["atr"].iloc[-1])

            print(f"✅ {symbol} | AI Score: {ai_score} | LTP: {ltp:.2f} | SL: {trailing_sl:.2f}")

            exit_qty = 0
            exit_reason = ""
            if ai_score >= 30:
                exit_qty = qty
                exit_reason = "Strong Exit Trigger"
            elif 15 <= ai_score < 30:
                exit_qty = int(qty / 2)
                exit_reason = "Partial Exit Triggered"

            all_rows.append([
                timestamp, symbol, qty, avg_price, ltp, trailing_sl, new_sl,
                ai_score, ", ".join(reasons) if reasons else "Holding"
            ])

            if exit_qty > 0 and is_market_open():
                kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange="NSE",
                    tradingsymbol=symbol,
                    transaction_type=kite.TRANSACTION_TYPE_SELL,
                    quantity=exit_qty,
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_CNC
                )
                pnl = (ltp - avg_price) * exit_qty
                pnl_msg = f"P&L: ₹{pnl:.2f} ({pnl / (avg_price * exit_qty) * 100:.2f}%)"
                send_telegram(BOT_TOKEN, CHAT_ID, f"⚠️ <b>{exit_reason}</b>\\n{symbol} | Qty: {exit_qty} | LTP: {ltp} | {pnl_msg} | Reasons: {', '.join(reasons)}")
                outcome = 1 if pnl > 0 else 0
                log_trade_to_sheet(log_sheet, timestamp, symbol, exit_qty, avg_price, ltp, "", "", "", ai_score, "SELL", exit_reason, pnl, outcome)

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
