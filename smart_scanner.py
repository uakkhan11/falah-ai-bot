# smart_scanner.py

import os
import json
import pandas as pd
import glob
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from credentials import get_kite

HIST_DIR = "/root/falah-ai-bot/historical_data/"

def load_all_live_prices():
    live = {}
    for f in glob.glob("/tmp/live_prices_*.json"):
        with open(f) as fd:
            live.update(json.load(fd))
    return live

import joblib

# Load model once
model = joblib.load("/root/falah-ai-bot/model.pkl")

score = 0
reasons = []

# Weighted heuristic
if ltp > last_daily["SMA20"]:
    score += 1.5
    reasons.append("Above SMA20")

if last_daily["EMA10"] > last_daily["EMA21"]:
    score += 1.2
    reasons.append("EMA10 > EMA21")

if last_daily["RSI"] and last_daily["RSI"] > 55:
    score += 0.8
    reasons.append(f"RSI {last_daily['RSI']:.1f}")

if atr and atr > 1.0:
    score += 1.0
    reasons.append(f"ATR {atr:.2f}")

if last_daily["volume"] > 1.2 * daily_df["volume"].rolling(10).mean().iloc[-1]:
    score += 2.0
    reasons.append("Volume breakout")

prev_close = daily_df["close"].iloc[-2]
today_open = last_daily["open"]
if today_open > prev_close * 1.02:
    score += 1.5
    reasons.append("Gap up")

# ML model score
import joblib
model = joblib.load("/root/falah-ai-bot/model.pkl")
features = [[
    last_daily["RSI"],
    last_daily["EMA10"],
    last_daily["EMA21"],
    last_daily["SMA20"],
    atr,
    last_daily["volume"] / daily_df["volume"].rolling(10).mean().iloc[-1]
]]
proba = model.predict_proba(features)[0][1]
ai_score = proba * 5.0  # Scale as desired
score += ai_score
reasons.append(f"AI Score {ai_score:.2f}")

# Optional sentiment score
try:
    from sentiment import get_sentiment_score
    sentiment = get_sentiment_score(sym)
    score += sentiment
    reasons.append(f"Sentiment {sentiment:+.2f}")
except:
    pass

print(f"👉 {sym} | Score: {score:.2f} | Reasons: {reasons}")


        # Compute indicators
        daily_df["SMA20"] = daily_df["close"].rolling(20).mean()
        daily_df["EMA10"] = EMAIndicator(close=daily_df["close"], window=10).ema_indicator()
        daily_df["EMA21"] = EMAIndicator(close=daily_df["close"], window=21).ema_indicator()
        daily_df["RSI"] = RSIIndicator(close=daily_df["close"], window=14).rsi()
        atr = AverageTrueRange(
            high=daily_df["high"],
            low=daily_df["low"],
            close=daily_df["close"],
            window=14
        ).average_true_range().iloc[-1]

        last_daily = daily_df.iloc[-1]

        score = 0
        reasons = []

        # Simple signals
        if ltp > last_daily["SMA20"]:
            score += 1
            reasons.append("Above SMA20")

        if last_daily["EMA10"] > last_daily["EMA21"]:
            score += 1
            reasons.append("EMA10 > EMA21")

        if last_daily["RSI"] and last_daily["RSI"] > 55:
            score += 1
            reasons.append(f"RSI {last_daily['RSI']:.1f}")

        if atr and atr > 1.0:
            score += 1
            reasons.append(f"ATR {atr:.2f}")

        if last_daily["volume"] > 1.2 * daily_df["volume"].rolling(10).mean().iloc[-1]:
            score += 1
            reasons.append("Volume breakout")

        prev_close = daily_df["close"].iloc[-2]
        today_open = last_daily["open"]
        if today_open > prev_close * 1.02:
            score += 1
            reasons.append("Gap up")

        print(f"👉 {sym} | Score: {score} | Reasons: {reasons}")
        
        results.append({
            "Symbol": sym,
            "CMP": ltp,
            "Score": score,
            "Reasons": ", ".join(reasons)
        })

    df = pd.DataFrame(results)

    if df.empty:
        print("⚠️ No stocks matched ANY criteria.")
        return df

    df = df.sort_values(by="Score", ascending=False)
    print("✅ Final scan results:")
    print(df)
    try:
        from sheets import log_scan_to_sheet
        log_scan_to_sheet(df)
    except Exception as e:
        print(f"⚠️ Failed to save to sheet: {e}")

    try:
        from utils import send_telegram
        msg = "🟢 Scan Results:\n" + "\n".join(
            f"{r['Symbol']} | {r['Score']} | {r['Reasons']}" for _, r in df.iterrows()
        )
        send_telegram(msg)
    except Exception as e:
        print(f"⚠️ Failed to send Telegram: {e}")
        
    return df

    # Save to Sheets and Telegram
    try:
        from sheets import log_scan_to_sheet
        log_scan_to_sheet(df)
    except Exception as e:
        print(f"⚠️ Failed to save to sheet: {e}")

    try:
        from utils import send_telegram
        msg = "🟢 Scan Results:\n" + "\n".join(
            f"{r['Symbol']} | {r['Score']} | {r['Reasons']}" for _, r in df.iterrows()
        )
        send_telegram(msg)
    except Exception as e:
        print(f"⚠️ Failed to send Telegram: {e}")

    return df
