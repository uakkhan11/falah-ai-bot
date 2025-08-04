# smart_scanner.py

import os
import pandas as pd
from indicators import (
    calculate_rsi, calculate_ema,
    detect_bullish_pivot, detect_macd_bullish_cross,
    detect_supertrend_green, calculate_bollinger_bands
)
from ai_engine import compute_ai_score
from holdings import get_existing_holdings
from live_price_reader import get_symbol_price_map
from credentials import load_secrets
import gspread

DATA_DIR = "/root/falah-ai-bot/historical_data"

def get_halal_list(sheet_key):
    gc = gspread.service_account(filename="/root/falah-ai-bot/falah-credentials.json")
    sheet = gc.open_by_key(sheet_key)
    ws = sheet.worksheet("HalalList")
    symbols = ws.col_values(1)[1:]  # skip header
    return [s.strip().upper() for s in symbols if s.strip()]

def run_smart_scan():
    secrets = load_secrets()
    symbols = get_halal_list(secrets["google"]["spreadsheet_key"])
    holdings = get_existing_holdings()
    live_prices = get_symbol_price_map()

    print(f"✅ Loaded {len(symbols)} symbols from Halal list")

    if not live_prices:
        print("⚠️ No live prices available. Possibly market is closed.")
        return pd.DataFrame(), {}

    final_selected = []
    skip_reasons = {}
    skip_reason_dict = {}
    filter_stats = {
        "ema_pass": 0,
        "pivot_pass": 0,
        "rsi_pass": 0,
        "supertrend_pass": 0,
        "macd_pass": 0
    }

    for symbol in sorted(set(symbols)):
        if symbol in holdings:
            skip_reasons["Holdings"] = skip_reasons.get("Holdings", 0) + 1
            skip_reason_dict[symbol] = "Already in holdings"
            continue

        if symbol not in live_prices:
            skip_reasons["No live price"] = skip_reasons.get("No live price", 0) + 1
            skip_reason_dict[symbol] = "No live price"
            continue

        filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(filepath):
            skip_reasons["Missing file"] = skip_reasons.get("Missing file", 0) + 1
            skip_reason_dict[symbol] = "Missing historical data file"
            continue

        df = pd.read_csv(filepath)
        if df.shape[0] < 30:
            skip_reasons["Not enough data"] = skip_reasons.get("Not enough data", 0) + 1
            skip_reason_dict[symbol] = "Less than 30 candles"
            continue

        df = df.tail(60).copy()
        df["rsi"] = calculate_rsi(df["close"])
        df["ema10"] = calculate_ema(df["close"], span=10)
        df["ema21"] = calculate_ema(df["close"], span=21)

        rsi = df["rsi"].iloc[-1]
        prev_rsi = df["rsi"].iloc[-2]
        ema10 = df["ema10"].iloc[-1]
        ema21 = df["ema21"].iloc[-1]

        if ema10 <= ema21:
            skip_reasons["EMA10 < EMA21"] = skip_reasons.get("EMA10 < EMA21", 0) + 1
            skip_reason_dict[symbol] = f"EMA10 ({ema10:.2f}) < EMA21 ({ema21:.2f})"
            continue
        filter_stats["ema_pass"] += 1

        if rsi < 30 or rsi > 75:
            skip_reasons[f"RSI {round(rsi,2)} out of 30-75"] = skip_reasons.get(f"RSI {round(rsi,2)} out of 30-75", 0) + 1
            skip_reason_dict[symbol] = f"RSI {rsi:.2f} out of range"
            continue
        if rsi < prev_rsi:
            skip_reasons["RSI falling"] = skip_reasons.get("RSI falling", 0) + 1
            skip_reason_dict[symbol] = f"RSI {rsi:.2f} falling from {prev_rsi:.2f}"
            continue
        filter_stats["rsi_pass"] += 1

        if not detect_bullish_pivot(df):
            skip_reasons["No bullish pivot"] = skip_reasons.get("No bullish pivot", 0) + 1
            skip_reason_dict[symbol] = "No bullish pivot"
            continue
        filter_stats["pivot_pass"] += 1

        if not detect_macd_bullish_cross(df):
            skip_reasons["No MACD bullish cross"] = skip_reasons.get("No MACD bullish cross", 0) + 1
            skip_reason_dict[symbol] = "No MACD bullish crossover"
            continue
        filter_stats["macd_pass"] += 1

        if not detect_supertrend_green(df):
            skip_reasons["Supertrend not green"] = skip_reasons.get("Supertrend not green", 0) + 1
            skip_reason_dict[symbol] = "Supertrend not green"
            continue
        filter_stats["supertrend_pass"] += 1

        df["bb_upper"], df["bb_middle"], df["bb_lower"] = calculate_bollinger_bands(df["close"])

        ltp = live_prices[symbol]
        bb_lower = df["bb_lower"].iloc[-1]
        bb_upper = df["bb_upper"].iloc[-1]

        # ✅ Bollinger Band filter: near lower band (within 5%)
        if ltp > bb_lower * 1.05:
            skip_reasons["Not near lower BB"] = skip_reasons.get("Not near lower BB", 0) + 1
            skip_reason_dict[symbol] = f"LTP {ltp:.2f} not near lower BB {bb_lower:.2f}"
            continue

        ltp = live_prices[symbol]
        ai_score, ai_reasons = compute_ai_score(df)

        final_selected.append({
            "symbol": symbol,
            "ltp": ltp,
            "Score": round(ai_score, 4),
            "rsi": round(rsi, 2),
            "ai_reasons": ", ".join(ai_reasons) if ai_reasons else ""
        })

    result_df = pd.DataFrame(final_selected)
    result_df.to_json("final_screened.json", orient="records", indent=2)

    print(f"\n✅ After filters: {len(result_df)} passed\n")
    for s in sorted(skip_reason_dict):
        print(f"⛔ Skipped {s}: Reason - {skip_reason_dict[s]}")

    return result_df, {
        "skip_reasons": skip_reasons,
        "filter_stats": filter_stats
    }

if __name__ == "__main__":
    result_df, debug_stats = run_smart_scan()

    if result_df.empty:
        print("✅ Final selected 0 stocks.")
    else:
        print(f"✅ Final selected {len(result_df)} stocks:")
        print(result_df[["symbol", "ltp", "Score", "rsi"]])

    # 📊 Skip Reason Summary
    if "skip_reasons" in debug_stats:
        print("\n📊 Skip Reason Summary:")
        for reason, count in debug_stats["skip_reasons"].items():
            print(f" - {reason}: {count}")

    # 📈 Filter Pass Stats
    if "filter_stats" in debug_stats:
        print("\n📈 Filter Pass Stats:")
        for key, val in debug_stats["filter_stats"].items():
            print(f" - {key}: {val}")
