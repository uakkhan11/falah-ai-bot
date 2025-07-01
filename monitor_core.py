# monitor_core.py

import time
import json
import pytz
import gspread
from datetime import datetime

from utils import get_cnc_holdings, send_telegram, analyze_exit_signals
from indicators import (
    calculate_atr_trailing_sl,
    check_supertrend_flip,
    check_rsi_bearish_divergence,
    check_vwap_cross,
)
from sheets import log_exit_to_sheet
from holdings_state import load_previous_exits, update_exit_log

IST = pytz.timezone("Asia/Kolkata")

def monitor_once(kite, token_map, log, live_prices):
    """
    Runs one monitoring cycle.
    """
    now = datetime.now(IST)
    market_open = now.weekday() < 5 and (
        (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
        and (now.hour < 15 or (now.hour == 15 and now.minute < 30))
    )
    today_str = now.strftime("%Y-%m-%d")

    try:
        holdings = get_cnc_holdings(kite)
    except Exception as e:
        log(f"❌ Error fetching CNC holdings: {e}")
        return

    if not holdings:
        log("❌ No CNC holdings found.")
        return

    # Load exited log
    exited = load_previous_exits("/root/falah-ai-bot/exited_stocks.json")
    if not isinstance(exited, dict):
        log("⚠️ exited_stocks.json invalid format, resetting.")
        exited = {}

    # Google Sheets connection
    gc = gspread.service_account(filename="/root/falah-credentials.json")
    sheet = gc.open_by_key("1ccAxmGmqHoSAj9vFiZIGuV2wM6KIfnRdSebfgx1Cy_c")
    monitor_tab = sheet.worksheet("MonitoredStocks")
    existing_rows = monitor_tab.get_all_records()

    for stock in holdings:
        symbol = stock.get("tradingsymbol") or stock.get("symbol")
        quantity = stock.get("quantity")
        avg_price = stock.get("average_price")

        token = token_map.get(symbol)
        if token is None:
            log(f"⚠️ No token for {symbol}. Skipping.")
            continue

        cmp = live_prices.get(int(token))
        if not cmp:
            log(f"⚠️ No live CMP for {symbol}. Skipping.")
            continue

        exposure = round(cmp * quantity, 2)

        # Already exited?
        if exited.get(symbol) == today_str:
            log(f"🔁 {symbol} already exited today. Skipping.")
            continue

        # Exit conditions
        sl_price = calculate_atr_trailing_sl(kite, symbol, cmp)
        sl_hit = sl_price and cmp <= sl_price

        try:
            st_flip_daily = check_supertrend_flip(kite, symbol)
        except Exception as e:
            log(f"⚠️ Supertrend daily check failed: {e}")
            st_flip_daily = False

        try:
            st_flip_15m = check_supertrend_flip(kite, symbol)
        except Exception as e:
            log(f"⚠️ Supertrend 15m check failed: {e}")
            st_flip_15m = False

        try:
            rsi_div = check_rsi_bearish_divergence(kite, symbol)
        except Exception as e:
            log(f"⚠️ RSI divergence check failed: {e}")
            rsi_div = False

        try:
            vwap_cross = check_vwap_cross(kite, symbol)
        except Exception as e:
            log(f"⚠️ VWAP cross check failed: {e}")
            vwap_cross = False

        ai_exit = analyze_exit_signals(symbol, avg_price, cmp)

        reasons = []
        if sl_hit:
            reasons.append(f"ATR SL hit ({sl_price})")
        if st_flip_daily and st_flip_15m:
            reasons.append("Supertrend flip")
        if rsi_div:
            reasons.append("RSI divergence")
        if vwap_cross:
            reasons.append("VWAP cross")
        if ai_exit:
            reasons.append("AI exit")

        if reasons:
            reason_str = ", ".join(reasons)
            log(f"🚨 Exit triggered for {symbol}: {reason_str}")

            # AUTO SELL ORDER
            if market_open:
                try:
                    kite.place_order(
                        variety=kite.VARIETY_REGULAR,
                        exchange=kite.EXCHANGE_NSE,
                        tradingsymbol=symbol,
                        transaction_type=kite.TRANSACTION_TYPE_SELL,
                        quantity=quantity,
                        order_type=kite.ORDER_TYPE_MARKET,
                        product=kite.PRODUCT_CNC
                    )
                    log(f"✅ Sell order placed for {symbol}.")
                except Exception as e:
                    log(f"⚠️ Sell order failed for {symbol}: {e}")
            else:
                log("⏸️ Market closed, sell not placed.")

            # Update exit log and notify
            update_exit_log("/root/falah-ai-bot/exited_stocks.json", symbol)
            send_telegram(
                f"🚨 Auto Exit Executed\n"
                f"Symbol: {symbol}\nPrice: {cmp}\nReasons: {reason_str}"
            )
            log_exit_to_sheet(
                "MonitoredStocks",
                "MonitoredStocks",
                symbol,
                cmp,
                reason_str
            )
        else:
            log(f"✅ {symbol}: Holding.")

