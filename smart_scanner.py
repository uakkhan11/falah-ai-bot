# smart_scanner.py

import os
import json
import pandas as pd
from indicators import add_all_indicators, detect_macd_bullish_cross, detect_supertrend_green

DATA_DIR = "historical_data/"
EXCLUDED_SYMBOLS = set()

def load_large_midcap_symbols():
    try:
        with open("large_mid_cap.json", "r") as f:
            return json.load(f)
    except:
        return []

def load_holdings_positions():
    try:
        with open("holdings_positions.json", "r") as f:
            return json.load(f)
    except:
        return []

def detect_bullish_pivot_relaxed(df):
    if len(df) < 5:
        return False
    lows = df["low"].tail(3).values
    closes = df["close"].tail(3).values
    return (lows[1] > lows[0]) and (lows[2] > lows[1]) and (closes[2] > closes[1])

def run_smart_scan():
    symbols = [f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    large_midcap = set(load_large_midcap_symbols())
    holdings = set(load_holdings_positions())
    EXCLUDED_SYMBOLS.update(holdings)

    filtered_symbols = [s for s in symbols if s in large_midcap and s not in EXCLUDED_SYMBOLS]
    print(f"✅ Loaded {len(symbols)} total symbols")
    print(f"✅ Loaded {len(large_midcap)} Large/Mid Cap symbols.")
    print(f"🚫 Skipping {len(holdings)} existing holdings/positions.")

    selected = []
    skip_reasons = {}
    filter_stats = {
        "ema_pass": 0,
        "pivot_pass": 0,
        "rsi_pass": 0,
        "supertrend_pass": 0,
        "macd_pass": 0
    }

    for symbol in filtered_symbols:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, f"{symbol}.csv"))
            if len(df) < 30:
                continue
            df = add_all_indicators(df)

            reasons = []
            latest = df.iloc[-1]
            ema10 = latest["EMA10"]
            ema21 = latest["EMA21"]
            rsi = latest["RSI"]

            # Relaxed EMA10 > EMA21 with 2% tolerance
            if ema10 < ema21 * 0.98:
                reasons.append("EMA10 below EMA21")
            else:
                filter_stats["ema_pass"] += 1

            # Relaxed bullish pivot
            if not detect_bullish_pivot_relaxed(df.tail(10)):
                reasons.append("No bullish pivot")
            else:
                filter_stats["pivot_pass"] += 1

            # RSI in range
            if not (30 <= rsi <= 75):
                reasons.append(f"RSI {rsi:.2f} out of range (30-75)")
            else:
                filter_stats["rsi_pass"] += 1

            # Optional: Supertrend confirmation
            if not detect_supertrend_green(df.tail(30)):
                reasons.append("Supertrend not green")
            else:
                filter_stats["supertrend_pass"] += 1

            # Optional: MACD crossover
            if not detect_macd_bullish_cross(df.tail(50)):
                reasons.append("No MACD bullish cross")
            else:
                filter_stats["macd_pass"] += 1

            if not reasons:
                selected.append({
                    "symbol": symbol,
                    "close": latest["close"],
                    "RSI": round(rsi, 2),
                    "EMA10": round(ema10, 2),
                    "EMA21": round(ema21, 2),
                })
            else:
                for r in reasons:
                    skip_reasons[r] = skip_reasons.get(r, 0) + 1

        except Exception as e:
            skip_reasons[f"{symbol} error"] = skip_reasons.get(f"{symbol} error", 0) + 1
            continue

    selected_df = pd.DataFrame(selected)
    print(f"\n✅ Final selected {len(selected_df)} stocks.")
    print(selected_df)

    # Detailed skip reason summary
    print("\n📊 Skip Reason Summary:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f" - {reason}: {count}")

    # Optional: Filter pass stats
    print("\n📈 Filter Pass Stats:")
    for k, v in filter_stats.items():
        print(f" - {k}: {v}")

    return selected_df

if __name__ == "__main__":
    run_smart_scan()
