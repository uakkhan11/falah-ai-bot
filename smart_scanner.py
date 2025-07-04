# smart_scanner.py

import os
import json
import pandas as pd
import glob
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
from utils import get_halal_list
from credentials import get_kite
from datetime import datetime, timedelta

HIST_DIR = "/root/falah-ai-bot/historical_data/"

def load_all_live_prices():
    live = {}
    for f in glob.glob("/tmp/live_prices_*.json"):
        with open(f) as fd:
            live.update(json.load(fd))
    return live

def run_smart_scan():
    kite = get_kite()
    live_prices = load_all_live_prices()
    with open("/root/falah-ai-bot/tokens.json") as f:
        token_map = json.load(f)
    token_to_symbol = {v: k for k, v in token_map.items()}

    results = []
    for token, ltp in live_prices.items():
        sym = token_to_symbol.get(str(token))
        if not sym:
            continue

        # Daily historical
        daily_file = os.path.join(HIST_DIR, f"{sym}.csv")
        if not os.path.exists(daily_file):
            continue

        daily_df = pd.read_csv(daily_file)
        if len(daily_df) < 21:
            continue

        # Compute Daily indicators
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

        # Multi-timeframe 15min data
        try:
            to_dt = datetime.now()
            from_dt = to_dt - timedelta(days=5)
            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_dt,
                to_date=to_dt,
                interval="15minute"
            )
            tf_df = pd.DataFrame(candles)
            tf_df["SMA20"] = tf_df["close"].rolling(20).mean()
        except Exception as e:
            print(f"⚠️ Could not fetch 15m for {sym}: {e}")
            continue

        last_15m = tf_df.iloc[-1]

        score = 0
        reasons = []

        # ✅ SMA(20)
        if ltp > last_daily["SMA20"]:
            score += 1
            reasons.append("Above SMA20")

        # ✅ EMA crossover
        if last_daily["EMA10"] > last_daily["EMA21"]:
            score += 1
            reasons.append("EMA10 > EMA21")

        # ✅ RSI filter
        if last_daily["RSI"] and last_daily["RSI"] > 55:
            score += 1
            reasons.append(f"RSI {last_daily['RSI']:.1f}")

        # ✅ ATR filter
        if atr and atr > 1.0:
            score += 1
            reasons.append(f"ATR {atr:.2f}")

        # ✅ Volume breakout
        if last_daily["volume"] > 1.2 * daily_df["volume"].rolling(10).mean().iloc[-1]:
            score += 1
            reasons.append("Volume breakout")

        # ✅ Gap up detection
        prev_close = daily_df["close"].iloc[-2]
        today_open = last_daily["open"]
        if today_open > prev_close * 1.02:
            score += 1
            reasons.append("Gap up")

        # ✅ Multi-timeframe confirmation
        if last_15m["close"] > last_15m["SMA20"]:
            score += 1
            reasons.append("15m SMA confluence")

        if score >= 3:
            results.append({
                "Symbol": sym,
                "CMP": ltp,
                "Score": score,
                "Reasons": ", ".join(reasons)
            })

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
    if not df.empty:
        # Save to Google Sheets (Phase 3)
        try:
            from sheets import log_scan_to_sheet
            log_scan_to_sheet(df)
        except Exception as e:
            print(f"⚠️ Failed to save to sheet: {e}")

        # Send Telegram (Phase 2)
        try:
            from utils import send_telegram
            msg = "🟢 Scan Results:\n" + "\n".join(
                f"{r['Symbol']} | {r['Score']} | {r['Reasons']}" for _, r in df.iterrows()
            )
            send_telegram(msg)
        except Exception as e:
            print(f"⚠️ Failed to send Telegram: {e}")

    return df
