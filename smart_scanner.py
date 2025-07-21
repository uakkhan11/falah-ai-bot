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
from indicators import detect_bullish_pivot

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
        print("⚠️ Live prices not found. Loading last close prices from historical files.")
        with open("/root/falah-ai-bot/tokens.json") as f:
            tokens = json.load(f)
        for sym in tokens.keys():
            daily_file = os.path.join(HIST_DIR, f"{sym}.csv")
            if os.path.exists(daily_file):
                df = pd.read_csv(daily_file)
                if not df.empty:
                    last_close = df.iloc[-1]["close"]
                    token = str(tokens[sym])
                    live[token] = last_close
        print(f"✅ Loaded {len(live)} fallback prices from historical files.")
    return live

def get_current_holdings_positions_symbols(kite):
    symbols = set()
    try:
        positions = kite.positions()
        holdings = kite.holdings()

        for p in positions['day'] + positions['net']:
            if p['quantity'] != 0:
                symbols.add(p['tradingsymbol'])
        for h in holdings:
            symbols.add(h['tradingsymbol'])

        print(f"🚫 Skipping {len(symbols)} stocks already in positions/holdings.")
    except Exception as e:
        print(f"⚠️ Error fetching holdings/positions: {e}")
    return symbols

def run_smart_scan():
    kite = get_kite()
    live_prices = load_all_live_prices()
    large_mid_symbols = load_large_mid_cap_symbols()

    with open("/root/falah-ai-bot/tokens.json") as f:
        tokens = json.load(f)
    token_to_symbol = {str(v): k for k, v in tokens.items()}

    skip_symbols = get_current_holdings_positions_symbols(kite)

    print(f"\n✅ Loaded {len(live_prices)} live prices.")

    if not live_prices:
        print("⚠️ No live prices available, exiting scan.")
        return pd.DataFrame()

    results = []
    for token, ltp in live_prices.items():
        sym = token_to_symbol.get(str(token))
        if not sym or sym in skip_symbols or sym not in large_mid_symbols:
            continue

        daily_file = os.path.join(HIST_DIR, f"{sym}.csv")
        if not os.path.exists(daily_file):
            continue

        daily_df = pd.read_csv(daily_file)
        if len(daily_df) < 21:
            continue

        daily_df["EMA10"] = EMAIndicator(close=daily_df["close"], window=10).ema_indicator()
        daily_df["EMA21"] = EMAIndicator(close=daily_df["close"], window=21).ema_indicator()
        daily_df["RSI"] = RSIIndicator(close=daily_df["close"], window=14).rsi()
        atr = AverageTrueRange(
            high=daily_df["high"], low=daily_df["low"], close=daily_df["close"], window=14
        ).average_true_range().iloc[-1]
        adx = ADXIndicator(
            high=daily_df["high"], low=daily_df["low"], close=daily_df["close"], window=14
        ).adx().iloc[-1]
        volume_change = daily_df["volume"].iloc[-1] / (daily_df["volume"].rolling(10).mean().iloc[-1] + 1e-9)

        last = daily_df.iloc[-1]
        rsi = last["RSI"]
        ema10 = last["EMA10"]
        ema21 = last["EMA21"]

        if not (40 <= rsi <= 70):
            print(f"❌ {sym} skipped due to RSI {rsi:.2f}")
            continue

        bullish_pivot = detect_bullish_pivot(daily_df.tail(30))
        if not bullish_pivot:
            print(f"❌ {sym} skipped due to no bullish pivot structure")
            continue

        features = [[rsi, ema10, ema21, atr, volume_change]]
        ai_score = model.predict_proba(features)[0][1] * 5

        if 65 < rsi <= 70 and ai_score < 1.5:
            print(f"❌ {sym} skipped RSI {rsi:.2f} AI {ai_score:.2f}")
            continue

        score = ai_score
        reasons = [f"AI Score {ai_score:.2f}"]

        if ema10 > ema21:
            score += 1.0
            reasons.append("EMA10 > EMA21")
        if volume_change > 1.2:
            score += 1.0
            reasons.append("Volume Spike")
        if atr > 1.0:
            score += 1.0
            reasons.append(f"ATR {atr:.2f}")
        if bullish_pivot:
            score += 1.0
            reasons.append("Bullish Pivot Detected")

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

        del daily_df
        gc.collect()

    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False)
    print(df)
    return df
