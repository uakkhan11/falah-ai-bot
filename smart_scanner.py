# smart_scanner.py

import os
import json
import pandas as pd
import glob
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange
from credentials import get_kite
import joblib
import gc
from indicators import (
    detect_bullish_pivot,
    detect_macd_bullish_cross,
    detect_supertrend_green
)

HIST_DIR = "/root/falah-ai-bot/historical_data/"
LARGE_MID_CAP_FILE = "/root/falah-ai-bot/large_mid_cap.json"
model = joblib.load("/root/falah-ai-bot/model.pkl")

def load_large_mid_cap_symbols():
    with open(LARGE_MID_CAP_FILE) as f:
        symbols = json.load(f)
    print(f"✅ Loaded {len(symbols)} Large/Mid Cap symbols.")
    return set(symbols)

def load_all_live_prices():
    live = {}
    files = glob.glob("/tmp/live_prices_*.json")
    if files:
        for f in files:
            with open(f) as fd:
                live.update(json.load(fd))
        print(f"✅ Loaded {len(live)} live prices.")
    else:
        print("⚠️ Live prices not found. Using last closes.")
        with open("/root/falah-ai-bot/tokens.json") as f:
            tokens = json.load(f)
        for sym in tokens.keys():
            daily_file = os.path.join(HIST_DIR, f"{sym}.csv")
            if os.path.exists(daily_file):
                df = pd.read_csv(daily_file)
                if not df.empty:
                    live[str(tokens[sym])] = df.iloc[-1]["close"]
        print(f"✅ Loaded {len(live)} fallback prices.")
    return live

def get_current_holdings_positions_symbols(kite):
    symbols = set()
    try:
        positions = kite.positions()
        holdings = kite.holdings()
        for p in positions['net']:
            if p['quantity'] != 0:
                symbols.add(p['tradingsymbol'])
        for h in holdings:
            symbols.add(h['tradingsymbol'])
        print(f"🚫 Skipping {len(symbols)} existing holdings/positions.")
    except Exception as e:
        print(f"⚠️ Error fetching positions: {e}")
    return symbols

def run_smart_scan():
    kite = get_kite()
    live_prices = load_all_live_prices()
    large_mid_symbols = load_large_mid_cap_symbols()

    with open("/root/falah-ai-bot/tokens.json") as f:
        tokens = json.load(f)
    token_to_symbol = {str(v): k for k, v in tokens.items()}
    skip_symbols = get_current_holdings_positions_symbols(kite)

    if not live_prices:
        print("❌ No live prices found. Exiting.")
        return pd.DataFrame()

    results = []

    for token, ltp in live_prices.items():
        sym = token_to_symbol.get(str(token))
        if not sym or sym in skip_symbols or sym not in large_mid_symbols:
            continue

        daily_file = os.path.join(HIST_DIR, f"{sym}.csv")
        if not os.path.exists(daily_file):
            continue

        df = pd.read_csv(daily_file)
        if len(df) < 30:
            continue

        df["EMA10"] = EMAIndicator(close=df["close"], window=10).ema_indicator()
        df["EMA21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
        df["RSI"] = RSIIndicator(close=df["close"], window=14).rsi()
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range().iloc[-1]
        adx = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx().iloc[-1]
        volume_change = df["volume"].iloc[-1] / (df["volume"].rolling(10).mean().iloc[-1] + 1e-9)

        last = df.iloc[-1]
        rsi, ema10, ema21 = last["RSI"], last["EMA10"], last["EMA21"]

        reasons = []
        skip = False

        if not (40 <= rsi <= 70):
            reasons.append(f"RSI {rsi:.2f} not in range")
            skip = True

        bullish_pivot = detect_bullish_pivot(df.tail(30))
        if not bullish_pivot:
            reasons.append("No bullish pivot")
            skip = True

        macd_cross = detect_macd_bullish_cross(df.tail(35))
        if not macd_cross:
            reasons.append("No MACD cross")
            skip = True

        supertrend_ok = detect_supertrend_green(df.tail(30))
        if not supertrend_ok:
            reasons.append("Supertrend not bullish")
            skip = True

        if skip:
            print(f"❌ {sym} skipped due to: {', '.join(reasons)}")
            continue

        features = [[rsi, ema10, ema21, atr, volume_change]]
        ai_score = model.predict_proba(features)[0][1] * 5

        score = ai_score
        reasons = [f"AI {ai_score:.2f}"]

        if ema10 > ema21:
            score += 1; reasons.append("EMA10>EMA21")
        if volume_change > 1.2:
            score += 1; reasons.append("Volume spike")
        if atr > 1.0:
            score += 1; reasons.append(f"ATR {atr:.2f}")

        results.append({
            "Symbol": sym,
            "CMP": ltp,
            "RSI": round(rsi, 2),
            "EMA10": round(ema10, 2),
            "EMA21": round(ema21, 2),
            "ATR": round(atr, 2),
            "ADX": round(adx, 2),
            "VolumeChange": round(volume_change, 2),
            "AI_Score": round(ai_score, 2),
            "Score": round(score, 2),
            "Reasons": ", ".join(reasons)
        })

        del df
        gc.collect()

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    print(df)
    return df

    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False)
    print(df)
    return df
