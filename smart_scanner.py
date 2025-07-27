# smart_scanner.py
import os
import pandas as pd
from indicators import (
    calculate_rsi, calculate_ema,
    detect_bullish_pivot, detect_macd_bullish_cross,
    detect_supertrend_green
)
from ai_engine import compute_ai_score
from price_fetcher import get_price
from amfi_large_midcap import load_large_midcap_symbols
from holdings import get_existing_holdings

DATA_DIR = "/root/falah-ai-bot/historical_data"

def run_smart_scan():
    large_midcap_symbols = load_large_midcap_symbols()
    holdings = get_existing_holdings()

    final_selected = []
    skip_reasons = {}
    filter_stats = {
        "ema_pass": 0,
        "pivot_pass": 0,
        "rsi_pass": 0,
        "supertrend_pass": 0,
        "macd_pass": 0
    }

    for symbol in sorted(large_midcap_symbols):
        if symbol in holdings:
            skip_reasons["Skipping existing holdings/positions"] = skip_reasons.get("Skipping existing holdings/positions", 0) + 1
            continue

        filepath = os.path.join(DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(filepath):
            skip_reasons["Missing historical file"] = skip_reasons.get("Missing historical file", 0) + 1
            continue

        df = pd.read_csv(filepath)
        if df.shape[0] < 30:
            skip_reasons["Not enough data"] = skip_reasons.get("Not enough data", 0) + 1
            continue

        df = df.tail(60).copy()
        df["rsi"] = calculate_rsi(df["close"])
        df["ema10"] = calculate_ema(df["close"], span=10)
        df["ema21"] = calculate_ema(df["close"], span=21)

        rsi = df["rsi"].iloc[-1]
        ema10 = df["ema10"].iloc[-1]
        ema21 = df["ema21"].iloc[-1]

        # Filter: EMA10 > EMA21
        if ema10 <= ema21:
            skip_reasons["EMA10 below EMA21"] = skip_reasons.get("EMA10 below EMA21", 0) + 1
            continue
        filter_stats["ema_pass"] += 1

        # Filter: RSI between 30–75
        if rsi < 30 or rsi > 75:
            skip_reasons[f"RSI {rsi:.2f} out of range (30-75)"] = skip_reasons.get(f"RSI {rsi:.2f} out of range (30-75)", 0) + 1
            continue
        filter_stats["rsi_pass"] += 1

        # Filter: Bullish pivot
        if not detect_bullish_pivot(df):
            skip_reasons["No bullish pivot"] = skip_reasons.get("No bullish pivot", 0) + 1
            continue
        filter_stats["pivot_pass"] += 1

        # Filter: MACD Bullish Cross
        if not detect_macd_bullish_cross(df):
            skip_reasons["No MACD bullish cross"] = skip_reasons.get("No MACD bullish cross", 0) + 1
            continue
        filter_stats["macd_pass"] += 1

        # Filter: Supertrend Green
        if not detect_supertrend_green(df):
            skip_reasons["Supertrend not green"] = skip_reasons.get("Supertrend not green", 0) + 1
            continue
        filter_stats["supertrend_pass"] += 1

        # ✅ Passed all filters, get live price and AI score
        ltp = get_price(symbol)
        ai_score = compute_ai_score(df)

        final_selected.append({
            "symbol": symbol,
            "ltp": ltp,
            "ai_score": round(ai_score, 4),
            "rsi": round(rsi, 2)
        })

    result_df = pd.DataFrame(final_selected)
    return result_df, {
        "skip_reasons": skip_reasons,
        "filter_stats": filter_stats
    }


if __name__ == "__main__":
    result_df, debug_stats = run_smart_scan()

    if result_df.empty:
        print("✅ Final selected 0 stocks.")
        print(result_df)
    else:
        print(f"✅ Final selected {len(result_df)} stocks:")
        print(result_df[["symbol", "ltp", "ai_score", "rsi"]])

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
