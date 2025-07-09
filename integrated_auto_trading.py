# integrated_auto_trading.py

import os
import json
import datetime
from kiteconnect import KiteConnect
import pandas as pd

# Import helpers
from data_fetch import fetch_historical_candles, get_live_ltp
from smart_scanner import run_smart_scan
from credentials import load_secrets
from my_confidence_functions import (
    compute_confidence_score,
    compute_allocation_weight,
    adjust_capital_based_on_confidence,
    compute_quantity
)
from trade_helpers import (
    compute_trailing_sl,
    is_market_open,
    send_telegram,
    log_trade_to_sheet
)

# ====== Load credentials ======
secrets = load_secrets()
API_KEY = secrets["zerodha"]["api_key"]
ACCESS_TOKEN = secrets["zerodha"]["access_token"]
BOT_TOKEN = secrets["telegram"]["bot_token"]
CHAT_ID = secrets["telegram"]["chat_id"]

# Initialize KiteConnect
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ====== Initialize Google Sheets ======
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("falah-credentials.json", scope)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(secrets["google"]["spreadsheet_key"])
log_sheet = sheet.worksheet("TradeLog")

# ====== Config ======
TOTAL_CAPITAL = 100000  # adjust as needed
MAX_TRADES = 5
DRY_RUN = False

# ====== Run Scanner ======
print("🔍 Running scanner...")
scan_df = run_smart_scan()
if scan_df.empty:
    print("❌ No signals found.")
    exit(0)

# ====== Select Top N by confidence ======
selected_stocks = []
for idx, row in scan_df.iterrows():
    symbol = row["Symbol"]
    try:
        # Historical data
        instrument_token = row["Token"] if "Token" in row else kite.ltp(f"NSE:{symbol}")[f"NSE:{symbol}"]["instrument_token"]
        hist_df = fetch_historical_candles(kite, instrument_token, interval="day", days=60)

        # Analyze indicators
        cmp = get_live_ltp(kite, symbol)
        analysis = {
            "adx": row.get("ADX", 25),
            "rsi": row.get("RSI", 50),
            "rel_strength": row.get("RelStrength", 1),
            "backtest_winrate": row.get("WinRate", 60)
        }
        confidence = compute_confidence_score(analysis)
        selected_stocks.append({
            "symbol": symbol,
            "token": instrument_token,
            "cmp": cmp,
            "confidence": confidence,
            "hist_df": hist_df
        })
    except Exception as e:
        print(f"❌ Skipping {symbol}: {e}")

# Sort by confidence
selected_stocks = sorted(selected_stocks, key=lambda x: x["confidence"], reverse=True)[:MAX_TRADES]

# ====== Process each selected stock ======
for stock in selected_stocks:
    try:
        symbol = stock["symbol"]
        cmp = stock["cmp"]
        confidence = stock["confidence"]
        hist_df = stock["hist_df"]

        print(f"\n✅ Processing {symbol} (Confidence: {confidence:.2f})")

        # Compute allocation
        weight = compute_allocation_weight(confidence)
        allocation = adjust_capital_based_on_confidence(TOTAL_CAPITAL, confidence)

        # Compute quantity
        qty = compute_quantity(allocation, cmp)
        if qty <= 0:
            print(f"❌ Skipping {symbol}: Invalid quantity {qty}")
            continue

        # Compute trailing stop
        trailing_sl = compute_trailing_sl(hist_df)

        # Market hours check
        if not is_market_open():
            send_telegram(BOT_TOKEN, CHAT_ID, f"Market closed. Skipping {symbol}.")
            continue

        # Place order
        if DRY_RUN:
            print(f"(Dry Run) Would place order: {symbol}, Qty: {qty}, CMP: ₹{cmp}")
            send_telegram(BOT_TOKEN, CHAT_ID, f"(Dry Run) Prepared order for {symbol}, Qty: {qty}")
        else:
            kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=kite.TRANSACTION_TYPE_BUY,
                quantity=qty,
                order_type=kite.ORDER_TYPE_MARKET,
                product=kite.PRODUCT_CNC
            )
            send_telegram(BOT_TOKEN, CHAT_ID, f"✅ Order placed for {symbol} Qty: {qty} CMP: ₹{cmp}\nTrailing SL: ₹{trailing_sl}")

        # Log to Google Sheets
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_trade_to_sheet(
            log_sheet,
            timestamp,
            symbol,
            qty,
            cmp,
            "Auto Trade Entry",
            f"Trailing SL: {trailing_sl}, Confidence: {confidence:.2f}"
        )

        print(f"✅ Completed {symbol}.")

    except Exception as e:
        msg = f"❌ Error processing {symbol}: {e}"
        print(msg)
        send_telegram(BOT_TOKEN, CHAT_ID, msg)

print("\n🎯 All trades processed.")
