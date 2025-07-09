# trade_helpers.py

import pandas as pd
from pytz import timezone
import datetime
import requests

# ========================
# ATR Calculation
# ========================
def calculate_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# ========================
# Compute Trailing Stop Loss
# ========================
def compute_trailing_sl(df, atr_multiplier=1.5, lookback=10):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    atr = calculate_atr(df).iloc[-1]
    highest_close = df["close"].rolling(lookback).max().iloc[-1]
    trailing_sl = highest_close - atr * atr_multiplier
    return round(trailing_sl, 2)

# ========================
# Market Hours Check
# ========================
def is_market_open():
    india = timezone("Asia/Kolkata")
    now = datetime.datetime.now(india)
    return (
        now.weekday() < 5 and
        now.hour >= 9 and
        (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
    )

# ========================
# Send Telegram Notification
# ========================
def send_telegram(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"[Telegram Error] {e}")

# ========================
# Log Trade to Google Sheet
# ========================
def log_trade_to_sheet(log_sheet, timestamp, symbol, qty, ltp, reason, extra=""):
    log_sheet.append_row([
        timestamp, symbol, "SELL", qty, ltp, reason, extra
    ])
