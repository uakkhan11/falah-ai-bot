# smart_scanner.py

updated_smart_scanner_code = '''
import pandas as pd
import os
import json
from indicators import (
    calculate_indicators,
    is_supertrend_green
)
from data_fetch import get_live_ltp
from utils import load_large_midcap_stocks, is_holding_stock

def run_smart_scan():
    print("🔍 Running smart scan...")

    # Load large and mid cap stock list
    large_midcap = load_large_midcap_stocks()
    print(f"✅ Loaded {len(large_midcap)} Large/Mid Cap symbols.")

    # Load fallback prices
    fallback_prices = {}
    if os.path.exists("fallback_prices.json"):
        with open("fallback_prices.json") as f:
            fallback_prices = json.load(f)
    else:
        print("⚠️ Live prices not found. Using last closes.")
    print(f"✅ Loaded {len(fallback_prices)} fallback prices.")

    # Get positions to skip
    positions = json.load(open("positions.json")) if os.path.exists("positions.json") else []
    print(f"🚫 Skipping {len(positions)} existing holdings/positions.")

    scanned_data = []
    skip_reasons = []

    for symbol in large_midcap:
        if symbol in positions:
            continue

        try:
            df = pd.read_csv(f"historical_data/{symbol}.csv")
        except Exception as e:
            continue

        if df.shape[0] < 50:
            continue

        df = calculate_indicators(df)
        last_row = df.iloc[-1]
        reasons = []

        # Supertrend green
        if not is_supertrend_green(df):
            reasons.append("Supertrend not green")

        # EMA filter
        if not (last_row["ema10"] > last_row["ema21"]):
            reasons.append("EMA10 below EMA21")

        # RSI filter
        rsi = last_row["rsi"]
        if rsi < 30 or rsi > 75:
            reasons.append(f"RSI {round(rsi, 2)} out of range (30-75)")

        if reasons:
            for r in reasons:
                skip_reasons.append(r)
            continue

        # Use live price if available
        ltp = get_live_ltp(symbol) or fallback_prices.get(symbol) or last_row["close"]

        scanned_data.append({
            "symbol": symbol,
            "ltp": ltp,
            "rsi": round(last_row["rsi"], 2),
            "ema10": round(last_row["ema10"], 2),
            "ema21": round(last_row["ema21"], 2),
            "supertrend": last_row["supertrend"]
        })

    # Output results
    result_df = pd.DataFrame(scanned_data)
    print(f"✅ Final selected {len(result_df)} stocks.")
    print(result_df)

    # Print summary of skip reasons
    print("\\n📊 Skip Reason Summary:")
    from collections import Counter
    for reason, count in Counter(skip_reasons).most_common():
        print(f" - {reason}: {count}")

    return result_df
'''

# Save to file
with open("/mnt/data/smart_scanner.py", "w") as f:
    f.write(updated_smart_scanner_code)

"/mnt/data/smart_scanner.py"

