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
# Compute Trailing Stop Loss (ATR-based or Entry-based)
# ========================
def compute_trailing_sl(entry_price=None, atr_value=None, df=None, atr_multiplier=1.5, lookback=10):
    """
    - If entry_price and atr_value are provided → compute based on them directly
    - If df is provided → compute ATR and highest close from df
    """
    if entry_price is not None and atr_value is not None:
        # Direct formula version
        return round(entry_price - (atr_multiplier * atr_value), 2)

    if df is not None:
        # DataFrame-based version
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        atr = calculate_atr(df).iloc[-1]
        highest_close = df["close"].rolling(lookback).max().iloc[-1]
        trailing_sl = highest_close - atr * atr_multiplier
        return round(trailing_sl, 2)

    raise ValueError("Either (entry_price & atr_value) or (df) must be provided to compute trailing SL.")

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
def log_trade_to_sheet(sheet, timestamp, symbol, qty, avg_price, ltp, *args):
    """
    - sheet: Google Sheet API object
    - args: extra fields (e.g., reason, notes, etc.)
    """
    try:
        sheet.append_row([timestamp, symbol, qty, avg_price, ltp] + list(args))
    except Exception as e:
        print(f"[Sheet Log Error] {e}")
