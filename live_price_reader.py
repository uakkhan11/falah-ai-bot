# live_price_reader.py

import os
import json
import glob
import time
from datetime import datetime, time as dt_time
import pytz

# ---------- Settings ----------
TMP_FOLDER = "/tmp"
TOKEN_MAP_PATH = "/root/falah-ai-bot/token_map.json"
MAX_FILE_AGE_SEC = 15  # ignore stale live_prices_*.json files older than 15 sec

IST = pytz.timezone("Asia/Kolkata")

# ---------- Market Time Check ----------
def is_market_open():
    now = datetime.now(IST).time()
    return dt_time(9, 15) <= now <= dt_time(15, 30)

# ---------- Load Token Map ----------
def load_token_map(path=TOKEN_MAP_PATH):
    with open(path) as f:
        return json.load(f)  # {symbol: token}

def reverse_token_map(token_map):
    return {v: k for k, v in token_map.items()}  # {token: symbol}

# ---------- Read Latest Live Price Files ----------
def get_combined_live_prices(tmp_folder=TMP_FOLDER, max_age_sec=MAX_FILE_AGE_SEC):
    prices = {}
    now = time.time()
    for path in glob.glob(os.path.join(tmp_folder, "live_prices_*.json")):
        try:
            modified = os.path.getmtime(path)
            if now - modified > max_age_sec:
                print(f"⚠️ Skipping stale file: {path}")
                continue
            with open(path) as f:
                worker_data = json.load(f)
                prices.update({int(k): v for k, v in worker_data.items()})
        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")
    return prices  # {token: price}

# ---------- Main: Get Symbol → Price Map ----------
def get_symbol_price_map():
    if not is_market_open():
        print("🔒 Market is closed. Skipping live price fetch.")
        return {}

    token_map = load_token_map()
    reverse_map = reverse_token_map(token_map)
    token_prices = get_combined_live_prices()
    
    return {
        reverse_map[token]: price
        for token, price in token_prices.items()
        if token in reverse_map
    }
